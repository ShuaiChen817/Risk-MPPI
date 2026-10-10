"""
@authors: Grady Williams, Autonomous Control and Decision Systems Laboratory, Georgia Institute of Technology, USA
          Ihab S. Mohamed, Vehicle Autonomy and Intelligence Lab, Indiana University, Bloomington, USA        
@brief: 完整的MPPI和log-MPPI控制器实现，适配TurtleBot3 ROS2
"""
import numpy as np
from scipy import interpolate
import scipy.signal
from jinja2 import Template
import warnings
import sys
import math

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from pycuda.curandom import XORWOWRandomNumberGenerator
    from pycuda import gpuarray
    CUDA_AVAILABLE = True
except ImportError:
    print("PyCUDA not available, using CPU fallback")
    CUDA_AVAILABLE = False

class TurtleBot3MPPI:
    """TurtleBot3的MPPI控制器，集成运动学模型和代价函数"""
    
    def __init__(self, state_dim, control_dim, num_samples, time_horizon, control_freq,
                 exploration_variance, robot_params, LogN_info, 
                 num_optimization_iterations=1, block_dim=(32, 1, 1), 
                 lambda_=1.0, cost_range=(-100000000, 10000000000)):
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.block_dim = block_dim
        self.num_samples = block_dim[0] * ((num_samples - 1) // self.block_dim[0] + 1)
        self.num_timesteps = int(time_horizon * control_freq)
        self.dt = 1.0 / control_freq

        self.grid_semantics = 'costmap'   # or 'riskmap'
        self.risk_cost_weight =400.0     #320.0  # 风险积分的权重，可调 180~320
        self.risk_collision_threshold = 85
        self.enable_near_penalty = True   # 仅 costmap 模式下启用“近障二次惩罚”

        
        # 机器人参数
        self.max_linear_vel = robot_params['max_linear_velocity']
        self.max_angular_vel = robot_params['max_angular_velocity']
        self.robot_radius = robot_params['robot_radius']
        self.collision_cost = robot_params['collision_cost']
        
        # log-MPPI参数
        self.dist_type, self.mu_LogN, self.std_LogN = LogN_info
        
        # 控制序列
        self.U = np.zeros((self.num_timesteps, self.control_dim), dtype=np.float32)
        self.last_U = np.zeros((self.num_timesteps, self.control_dim), dtype=np.float32)
        
        # 其他参数
        self.exploration_variance = exploration_variance
        self.num_optimization_iterations = num_optimization_iterations
        self.lambda_ = lambda_
        self.cost_range = cost_range
        
        # 地图数据
        self.obstacle_grid = None
        self.map_info = None
        self.obstacle_grid_d = None   # ← GPU 里的代价图（gpuarray），后面会上传

        
        # 地图参数新增
        self.occ_threshold = 50          # 栅格>50视为障碍
        self.best_traj = None            # 本轮最优轨迹(用于可视化/发布local_plan)

        # —— 避障&稳定 —— 
        self.inflation_radius = 0.35   # m，判定“靠近障碍物”的半径
        self.near_obs_weight  = 25.0   # 靠近障碍物的代价权重（可再加大到 40）
        self.rate_cost_weight = 0.15   # 控制输入变化率 |u_t - u_{t-1}|^2 的权重，抑制抖动
        

        # === 新增：把变化率惩罚分成“线速度”和“角速度”两路，角速度变化给更大权重 ===
        self.rate_cost_w_lin  = 0.04     # 线速度变化率权重（温和）
        self.rate_cost_w_ang  = 0.45     # 角速度变化率权重（强力抑制左右甩）

        
        # 采样轨迹可视化
        self.debug_samples = None        # 采样轨迹可视化
        self._debug_costs     = None   # 形状：(N,)       每条采样的总cost
        self._debug_best_idx  = -1     # 最优轨迹索引
        self.best_traj        = None   # (T+1,3)         选中用于跟踪的轨迹（绿色
        
                # —— 性能/稳定性开关（新增）——
        self.enable_near_penalty = True   # 先关掉近障软惩罚，加速 & 防止“8”字抖动
        self.near_margin = 0.35            # 软惩罚的缓冲带（m）
        self.near_gain = 300.0                # 软惩罚权重（可再调）

        # 距离场缓存
        self.lethal_threshold = 50           # 认为是占据的阈值（与 costmap 相同语义）
        self.dist_field = None               # np.ndarray[H,W]，单位：米
        self.map_info = None                 # {'width','height','resolution','origin_x','origin_y'}

        self.cpu_num_samples_cap = 48      # CPU 端的采样上限（别用太大）

        self.control_dim = 2
        # 名义控制序列 U（T×2）如果已有就保持，否则初始化
        if not hasattr(self, 'U'):
            self.U = np.zeros((self.num_timesteps, self.control_dim), dtype=np.float32)

        # ★ 新增：上一拍控制（用于连续性/重置）
        self.last_control = np.zeros((self.control_dim,), dtype=np.float32)



        
        
        # 滤波参数
        self.SG_window = 7
        self.SG_PolyOrder = 3
        self.debug_samples = None # 用于调试的样本轨迹
        
        # 初始化CUDA（如果可用）
        if CUDA_AVAILABLE:
            self.generator = XORWOWRandomNumberGenerator()
            self.init_cuda()
        else:
            print("使用CPU版本MPPI")


    # === 新增：设置语义/阈值 & 风险读取 ===
    def set_map_semantics(self, mode: str, risk_collision_threshold: int = 85):
        """设置当前栅格的语义:
        - mode='costmap': 使用占据阈值（50）做碰撞，保留近障软惩罚
        - mode='riskmap': 使用更高阈值做“极高风险=碰撞”，并在代价中积分风险，关闭近障软惩罚避免双计
        """
        mode = mode.lower().strip() if isinstance(mode, str) else 'costmap'
        if mode not in ('costmap', 'riskmap'):
            mode = 'costmap'
        self.grid_semantics = mode
        self.risk_collision_threshold = int(risk_collision_threshold)
        self.enable_near_penalty = (mode == 'costmap')
        #self.get_logger().info(f'MPPI语义更新: {mode}, 阈值={self.risk_collision_threshold}')
        print(f'[MPPI] grid_semantics={mode}, thr={self.risk_collision_threshold}')


    def _get_collision_threshold(self):
        return 50 if self.grid_semantics == 'costmap' else int(self.risk_collision_threshold)

    def _risk_value(self, x, y):
        """读取当前位置的风险值，返回 0..1。出界或无图返回 0。"""
        if self.obstacle_grid is None or self.map_info is None:
            return 0.0
        rc = self._world_to_grid_rc(x, y)
        if rc is None:
            return 0.0
        r, c = rc
        H, W = self.obstacle_grid.shape
        if r < 0 or r >= H or c < 0 or c >= W:
            return 0.0
        val = float(self.obstacle_grid[r, c])
        if val < 0:
            val = 0.0
        return max(0.0, min(1.0, val / 100.0))

    def init_cuda(self):
        """初始化CUDA函数"""
        try:
            # 编译CUDA代码
            cuda_code = self.generate_cuda_code()
            mod = SourceModule(cuda_code)
            
            # 获取CUDA函数
            self.rollout_kernel = mod.get_function("rollout_kernel")
            self.U_d_ptr = mod.get_global("U_d")[0]
            
            print("CUDA初始化成功")
        except Exception as e:
            print(f"CUDA初始化失败: {e}")
            global CUDA_AVAILABLE
            CUDA_AVAILABLE = False

    def update_costmap(self, costmap_data, map_info):
        """更新代价地图（统一存成 [H, W] 的二维数组，防止索引越界）"""
        arr = np.asarray(costmap_data, dtype=np.int16)
        w = int(map_info['width'])
        h = int(map_info['height'])
        try:
            # nav2 的 OccupancyGrid.data 是一维，长 = w*h
            if arr.ndim == 1:
                arr = arr.reshape(h, w)
            elif arr.shape != (h, w):
                # 如果外面传进来就是二维但维度反了，强制 reshape
                arr = arr.reshape(h, w)
        except Exception:
            # 防御：维度不匹配时直接展平再重塑
            arr = arr.ravel()[: w * h].reshape(h, w)

        self.obstacle_grid = arr  # 形状 [H, W]
        self.map_info = {
            'width': w,
            'height': h,
            'resolution': float(map_info['resolution']),
            'origin_x': float(map_info['origin_x']),
            'origin_y': float(map_info['origin_y'])
        }
        # === GPU 同步：把代价图传到显存（float32） ===
        if CUDA_AVAILABLE:
            from pycuda import gpuarray
            grid_f = self.obstacle_grid.astype(np.float32).ravel()
            if (self.obstacle_grid_d is None) or (int(self.obstacle_grid_d.size) != int(grid_f.size)):
                self.obstacle_grid_d = gpuarray.to_gpu(grid_f)
            else:
                self.obstacle_grid_d.set(grid_f)



    def compute_control(self, current_state, target_state, weights):
        """计算控制输入（CPU版本优先，保存最优轨迹 + 采样轨迹用于可视化）"""
        # 没地图就先别跑，避免不必要的假碰撞
        if self.obstacle_grid is None or self.map_info is None:
            return np.array([0.0, 0.0], dtype=np.float32), 0.0, float('inf')

        # 确保输入类型正确
        current_state = np.asarray(current_state, dtype=np.float32).copy()
        target_state  = np.asarray(target_state,  dtype=np.float32).copy()
        weights       = np.asarray(weights,       dtype=np.float32).copy()

        # 给采样一个“可行的名义控制”作为均值（允许边转边走）
        self._set_nominal_from_goal(current_state, target_state)

        # 采样的标准差、控制代价权重
        std_n = np.array(self.exploration_variance, dtype=np.float32)
        R = np.array([0.1, 0.1], dtype=np.float32)

        cost_params = (std_n, R, weights, target_state)

        normalizer, min_cost = self.rollouts(current_state, cost_params)

        # 第一个控制指令
        u = self.U[0, :].copy().astype(np.float32)

        # 滑动控制序列
        self.slide_control_sequence()

        return u, normalizer, min_cost


    def rollouts(self, state, cost_params):
        """执行轨迹采样和评估"""
        if CUDA_AVAILABLE and hasattr(self, 'rollout_kernel') and (self.obstacle_grid_d is not None):
            return self.cuda_rollouts(state, cost_params)
        else:
            return self.cpu_rollouts(state, cost_params)

    def cuda_rollouts(self, state, cost_params):
        from pycuda import gpuarray, driver as cuda
        import pycuda.cumath as cumath

        std_n, R, weights, targets = cost_params

        # 1) 生成噪声（标准正态 或 log-MPPI）
        n_elems = self.num_samples * self.num_timesteps * self.control_dim
        if self.dist_type == 0:
            du_d = self.generator.gen_normal(n_elems, np.float32)
        else:
            n1 = self.generator.gen_normal(n_elems, np.float32)
            n2 = self.generator.gen_normal(n_elems, np.float32)
            logn = cumath.exp(self.mu_LogN + self.std_LogN * n2)
            du_d = n1 * logn

        # 2) 把 U 拷到 constant: U_d
        U_h = self.U.astype(np.float32).ravel()
        cuda.memcpy_htod(self.U_d_ptr, U_h)   # host→constant

        # 3) 其余参数搬到 GPU
        state_d   = gpuarray.to_gpu(np.asarray(state,   dtype=np.float32))
        std_nd    = gpuarray.to_gpu(np.asarray(std_n,   dtype=np.float32))
        R_d       = gpuarray.to_gpu(np.asarray(R,       dtype=np.float32))
        weights_d = gpuarray.to_gpu(np.asarray(weights, dtype=np.float32))
        targets_d = gpuarray.to_gpu(np.asarray(targets, dtype=np.float32))
        costs_d   = gpuarray.zeros(self.num_samples, dtype=np.float32)

        assert self.obstacle_grid_d is not None and self.map_info is not None, "GPU: costmap 未上传"
        map_w = np.int32(self.map_info['width'])
        map_h = np.int32(self.map_info['height'])
        res   = np.float32(self.map_info['resolution'])
        ox    = np.float32(self.map_info['origin_x'])
        oy    = np.float32(self.map_info['origin_y'])

        # 4) 启动内核
        blocksize = self.block_dim
        gridsize  = ((self.num_samples - 1) // self.block_dim[0] + 1, 1, 1)
        self.rollout_kernel(
            costs_d, du_d,
            state_d, std_nd, R_d, weights_d, targets_d,
            self.obstacle_grid_d, map_w, map_h, res, ox, oy,
            np.float32(self.dt),
            np.float32(self.max_linear_vel),
            np.float32(self.max_angular_vel),
            np.float32(self.collision_cost),
            np.int32(self.num_samples),
            np.int32(self.num_timesteps),
            grid=gridsize, block=blocksize
        )
        cuda.Context.synchronize()

        # 5) 取回结果
        costs = costs_d.get()
        ctrl_noise = du_d.get().reshape((self.num_samples, self.num_timesteps, self.control_dim))

        # 6) 可视化样本（前 K 条）
        K = int(min(24, self.num_samples))
        dbg_trajs = []
        for i in range(K):
            U_i = self.U + ctrl_noise[i]
            s = np.asarray(state, dtype=np.float32).copy()
            traj = [s.copy()]
            for t in range(self.num_timesteps):
                s = self.turtlebot3_kinematics(s, U_i[t])
                traj.append(s.copy())
            dbg_trajs.append(np.stack(traj, axis=0))
        self.debug_samples = np.stack(dbg_trajs, axis=0) if len(dbg_trajs) > 0 else None

        # 7) 最优轨迹（绿色）
        best_idx = int(np.argmin(costs))
        U_best = self.U + ctrl_noise[best_idx]
        s = np.asarray(state, dtype=np.float32).copy()
        best_traj = [s.copy()]
        for t in range(self.num_timesteps):
            s = self.turtlebot3_kinematics(s, U_best[t])
            best_traj.append(s.copy())
        self.best_traj = np.stack(best_traj, axis=0)
        self._debug_costs = costs.copy()
        self._debug_best_idx = best_idx

        # 8) 软最小化更新（与 CPU 一致）
        min_cost = float(np.min(costs))
        exp_c = np.exp(-(1.0 / self.lambda_) * (costs - min_cost))
        normalizer = float(np.sum(exp_c))
        if normalizer > 1e-9 and np.isfinite(exp_c).all():
            w_norm = exp_c / normalizer           # (N,)
            update = np.tensordot(w_norm, ctrl_noise, axes=(0, 0))  # (T,2)
            self.U = self.U + 0.18 * update
            self.U = self.savitzky_golay_filter(self.U)
            self.U[:, 0] = np.clip(self.U[:, 0], -self.max_linear_vel, self.max_linear_vel)
            self.U[:, 1] = np.clip(self.U[:, 1], -self.max_angular_vel, self.max_angular_vel)

        return normalizer, min_cost

    def cpu_rollouts(self, state, cost_params):
        """CPU 采样（保存最优轨迹 + 部分样本轨迹用于可视化）"""
        std_n, R, weights, targets = cost_params

        N = int(min(self.num_samples, self.cpu_num_samples_cap))  # CPU上别太多
        T = int(self.num_timesteps)

        costs = np.empty(N, dtype=np.float32)
        ctrl_noise = np.empty((N, T, self.control_dim), dtype=np.float32)

        # 采样轨迹可视化：只保留前K条
        K = int(min(24, N))
        dbg_trajs = []

        for i in range(N):
            # 采样噪声：Normal 或 Normal*LogNormal（log-MPPI）
            if self.dist_type == 0:
                noise = np.random.normal(0.0, std_n, (T, self.control_dim)).astype(np.float32)
            else:
                normal = np.random.normal(0.0, std_n, (T, self.control_dim)).astype(np.float32)
                logn   = np.random.lognormal(self.mu_LogN, self.std_LogN, (T, self.control_dim)).astype(np.float32)
                noise  = normal * logn

            U_i = self.U + noise  # 围绕名义控制采样
            c = self.simulate_trajectory(state, U_i, targets, weights, R)
            costs[i] = c
            ctrl_noise[i] = noise

            if i < K:
                # 回放“名义+噪声”的状态序列
                s = np.asarray(state, dtype=np.float32).copy()
                traj = [s.copy()]
                for t in range(T):
                    s = self.turtlebot3_kinematics(s, U_i[t])
                    traj.append(s.copy())
                dbg_trajs.append(np.stack(traj, axis=0))  # (T+1, 3)

        # 记录可视化样本
        self.debug_samples = np.stack(dbg_trajs, axis=0) if len(dbg_trajs) > 0 else None

        # 软最小化
        min_cost = float(np.min(costs))
        exp_c = np.exp(-(1.0 / self.lambda_) * (costs - min_cost))
        normalizer = float(np.sum(exp_c))

        # 记下“当前采样中成本最低的那条”，用于发布local_plan
        best_idx = int(np.argmin(costs))
        U_best = self.U + ctrl_noise[best_idx]
        s = np.asarray(state, dtype=np.float32).copy()
        best_traj = [s.copy()]
        for t in range(T):
            s = self.turtlebot3_kinematics(s, U_best[t])
            best_traj.append(s.copy())
        self.best_traj = np.stack(best_traj, axis=0)  # (T+1,3)

        # 记录用于可视化的代价和最佳索引
        self._debug_costs = costs.copy()
        self._debug_best_idx = int(best_idx)


        # 按权重更新控制序列
        if normalizer > 1e-9 and np.isfinite(exp_c).all():
            w_norm = exp_c / normalizer  # (N,)
            update = np.tensordot(w_norm, ctrl_noise, axes=(0, 0))  # (T,2)
            # 学习率保守一点，避免大跳
            self.U = self.U + 0.18 * update

            # 平滑 + 限幅
            self.U = self.savitzky_golay_filter(self.U)
            self.U[:, 0] = np.clip(self.U[:, 0], -self.max_linear_vel, self.max_linear_vel)
            self.U[:, 1] = np.clip(self.U[:, 1], -self.max_angular_vel, self.max_angular_vel)

        return normalizer, min_cost



    def simulate_trajectory(self, initial_state, control_seq, targets, weights, R):
        """
        模拟单条轨迹并计算代价：
        - 状态/控制代价
        - 硬碰撞提前截断
        - 平滑近障二次惩罚（安全圈 + 缓冲带）
        - 变化率惩罚：线/角分别加权（特别压角速度抖动）
        - 曲率惩罚：高速+大曲率不鼓励（避免高速猛拐）
        """
        state = np.array(initial_state, dtype=np.float32).copy()
        total_cost = 0.0

        # 安全圈 + 缓冲带（米）
        safe_margin = 0.25
        safe_radius = float(self.robot_radius) + safe_margin
        near_bonus  = 0.7  #0.45
        near_gain   = 400.0     #360.0

        # 变化率惩罚的权重
        w_lin = float(getattr(self, "rate_cost_w_lin", 0.08))
        w_ang = float(getattr(self, "rate_cost_w_ang", 0.45))
        u_prev = np.array([0.0, 0.0], dtype=np.float32)

        # 曲率惩罚（|w| / max(v,eps) 太大时给罚）
        kappa_w = 0.6  # 权重 1.5~2.5 之间可调
        eps_v   = 0.06
        kappa_max = 0.8  # 允许的最大曲率（越小越保守）

        T = len(control_seq)
        for t in range(T):
            u = np.array(control_seq[t], dtype=np.float32)
            state = self.turtlebot3_kinematics(state, u)

            # 基本代价
            dx = float(state[0] - targets[0])
            dy = float(state[1] - targets[1])
            dth = self._angle_norm(float(state[2] - targets[2]))   # ← 关键
            state_cost = float(weights[0]*dx*dx + weights[1]*dy*dy + weights[2]*dth*dth)

            control_cost = float(np.sum(R * u**2))

            # 变化率惩罚（线/角分别加权）
            du_lin = u[0] - u_prev[0]
            du_ang = u[1] - u_prev[1]
            rate_cost = w_lin * float(du_lin * du_lin) + w_ang * float(du_ang * du_ang)
            u_prev = u

            # 硬碰撞：直接砍尾
            if self._disc_collision(state[0], state[1], safe_radius):
                total_cost += float(self.collision_cost) * (T - t)
                break

            # 近障二次惩罚（温和+平滑）
            d = self._nearest_obstacle_dist(state[0], state[1], search_radius=safe_radius + near_bonus)
            if d is not None and d < (safe_radius + near_bonus):
                gap = (safe_radius + near_bonus - d)
                state_cost += near_gain * (gap * gap)

            # 曲率惩罚：高速大转弯不鼓励（避免“猛拐撞下一个障碍”）
            v_abs = abs(float(u[0]))
            w_abs = abs(float(u[1]))
            if v_abs > 0.10:
                kappa = w_abs / max(v_abs, eps_v)
                if kappa > kappa_max:
                    state_cost += kappa_w * (kappa - kappa_max) * (kappa - kappa_max)


            # 风险积分（RiskMap 模式）
            if self.grid_semantics == 'riskmap':
                r = self._risk_value(state[0], state[1])   # 0..1
                state_cost += self.risk_cost_weight * r


            total_cost += state_cost + control_cost + rate_cost

           


        return total_cost


    
    def get_debug_samples(self, max_count=24):
        """返回用于可视化的采样轨迹，形状 (N, T, 3) 或 (N, T+1, 3)"""
        if self.debug_samples is None:
            return None
        return self.debug_samples[:max_count].copy()
    
    def get_best_trajectory(self):
        """返回本轮采样里成本最低的状态序列 (T+1,3)，给上层发布 /local_plan 用"""
        
        return None if self.best_traj is None else self.best_traj.copy()


    def turtlebot3_kinematics(self, state, control):
        """TurtleBot3运动学模型"""
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        v, w = float(control[0]), float(control[1])
        
        # 应用速度限制
        v = np.clip(v, -self.max_linear_vel, self.max_linear_vel)
        w = np.clip(w, -self.max_angular_vel, self.max_angular_vel)
        
        # 更新状态
        new_state = np.array([
            x + np.cos(theta) * v * self.dt,
            y + np.sin(theta) * v * self.dt,
            theta + w * self.dt
        ]).astype(np.float32)

        return new_state

    def is_collision(self, state):
        """基于二维网格的简单碰撞检测（大于50判为障碍）"""
        if self.obstacle_grid is None or self.map_info is None:
            return False

        x, y = float(state[0]), float(state[1])
        res = self.map_info['resolution']
        ox, oy = self.map_info['origin_x'], self.map_info['origin_y']
        H, W = self.obstacle_grid.shape

        mx = int((x - ox) / res)
        my = int((y - oy) / res)

        # 出界视为碰撞（更保守）
        if mx < 0 or mx >= W or my < 0 or my >= H:
            return True

        return self.obstacle_grid[my, mx] > 50
    
    def _grid_index(self, x_m, y_m):
        """世界坐标 -> 栅格索引；超界返回 None"""
        if self.obstacle_grid is None or self.map_info is None:
            return None
        res = self.map_info['resolution']
        ox  = self.map_info['origin_x']
        oy  = self.map_info['origin_y']
        w   = self.map_info['width']
        h   = self.map_info['height']
        gx  = int((x_m - ox) / res)
        gy  = int((y_m - oy) / res)
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return None
        return gx, gy
    
    def get_debug_costs_and_best(self):
        """
        返回 (costs, best_idx)；用于上层对采样轨迹按代价上色
        """
        if self._debug_costs is None or self._debug_best_idx < 0:
            return None, -1
        return self._debug_costs.copy(), int(self._debug_best_idx)


    def obstacle_nearness_cost(self, state):
        """
        简易“靠近障碍物”代价：在以 inflation_radius 为半径的邻域里
        找最近的占用格，按 (1 - d/R)^2 计惩罚。没有障碍则 0。
        """
        if self.obstacle_grid is None or self.map_info is None:
            return 0.0

        R = float(self.inflation_radius)
        res = self.map_info['resolution']
        step = max(1, int(0.05 / res))  # 5cm 采样一次
        idx = self._grid_index(state[0], state[1])
        if idx is None:
            return 0.0

        w = self.map_info['width']
        # 邻域半径(格)
        rad_c = int(R / res)
        gx, gy = idx
        best_d = None

        # 扫一圈邻域（稀疏点）
        for dy in range(-rad_c, rad_c + 1, step):
            yy = gy + dy
            if yy < 0 or yy >= self.map_info['height']:
                continue
            for dx in range(-rad_c, rad_c + 1, step):
                xx = gx + dx
                if xx < 0 or xx >= self.map_info['width']:
                    continue
                occ = int(self.obstacle_grid[yy, xx])
                if occ > 50:  # 认为是障碍
                    # 距离（米）
                    d = math.hypot(dx * res, dy * res)
                    if best_d is None or d < best_d:
                        best_d = d

        if best_d is None:
            return 0.0
        if best_d >= R:
            return 0.0
        # (1 - d/R)^2 平滑上升
        return (1.0 - best_d / R) ** 2



    def savitzky_golay_filter(self, control_update):
        L = len(control_update)
        if L < self.SG_window:
            return control_update
        """Savitzky-Golay滤波器平滑控制序列"""
        if len(control_update) < self.SG_window:
            return control_update
        
        filtered_update = np.zeros_like(control_update)
        for i in range(self.control_dim):
            filtered_update[:, i] = scipy.signal.savgol_filter(
                control_update[:, i], self.SG_window, self.SG_PolyOrder, mode='mirror')
        return filtered_update
    
    def reset_nominal(self):
        # 把名义控制序列 U 置零，避免沿用上一次的方向
        self.U[:] = 0.0
        self.last_control[:] = 0.0

    def clear_best_traj(self):
        self.best_traj = None

    
   
    @staticmethod
    def _angle_norm(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _world_to_grid_rc(self, x, y):
        """世界坐标->行列索引(row, col)"""
        res = self.map_info['resolution']
        ox, oy = self.map_info['origin_x'], self.map_info['origin_y']
        r = int((y - oy) / res)
        c = int((x - ox) / res)
        return r, c

    def _disc_collision(self, x, y, radius):
        """以机器人半径做圆盘碰撞（检查邻域格子是否>阈值）"""
        if self.obstacle_grid is None or self.map_info is None:
            return False
        H, W = self.obstacle_grid.shape
        res = self.map_info['resolution']
        rad_cells = max(1, int(radius / res))
        r0, c0 = self._world_to_grid_rc(x, y)
        for dr in range(-rad_cells, rad_cells + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H: 
                return True  # 出界视为碰撞（更保守）
            for dc in range(-rad_cells, rad_cells + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    return True
                if dr*dr + dc*dc <= rad_cells*rad_cells:
                    if int(self.obstacle_grid[rr, cc]) > int(self.occ_threshold):
                        return True
        return False

    def _nearest_obstacle_dist(self, x, y, search_radius):
        """近邻搜索最近障碍物距离（米）；找不到返回 None"""
        if self.obstacle_grid is None or self.map_info is None:
            return None
        H, W = self.obstacle_grid.shape
        res = self.map_info['resolution']
        rad_cells = max(1, int(search_radius / res))
        r0, c0 = self._world_to_grid_rc(x, y)

        best = None
        for dr in range(-rad_cells, rad_cells + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H: 
                continue
            for dc in range(-rad_cells, rad_cells + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                if int(self.obstacle_grid[rr, cc]) > int(self.occ_threshold):
                    d = res * math.hypot(dr, dc)
                    best = d if best is None else min(best, d)
        return best


    def _set_nominal_from_goal(self, current_state, target_state):
        """把 U 预填为一个面向目标的恒定（v,w），让采样更有效"""
        dx = float(target_state[0] - current_state[0])
        dy = float(target_state[1] - current_state[1])
        dist = float(np.hypot(dx, dy))

        ang_to_goal = float(np.arctan2(dy, dx))
        ang_err = self._angle_norm(ang_to_goal - float(current_state[2]))

        # 角度大时先慢一点走；角度小再提速
        base_v = min(self.max_linear_vel * 0.95, max(0.35, 0.85 * dist))
        v_pref = base_v * max(0.5, np.cos(min(abs(ang_err), np.pi/2)))

        # 角速度比例温和点，避免大幅度 overshoot
        w_pref = np.clip(1.1 * ang_err, -self.max_angular_vel, self.max_angular_vel)

        self.U[:, 0] = v_pref
        self.U[:, 1] = w_pref


    def slide_control_sequence(self):
        """滑动控制序列"""
        U_new = np.zeros_like(self.U)
        U_new[:-1, :] = self.U[1:, :]
        U_new[-1, :] = 0.0  # 最后一个控制设为0
        self.U = U_new

    def to_gpu(self, array):
        """转换数组到GPU"""
        if not CUDA_AVAILABLE:
            return array
        array_flat = array.flatten().astype(np.float32)
        return gpuarray.to_gpu(array_flat)

    def generate_cuda_code(self):
        from jinja2 import Template
        cuda_template = Template(r"""
        #include <math.h>

        __device__ void turtlebot3_kinematics(float* state, const float* control, float dt,
                                            float max_v, float max_w) {
            float v = fminf(max_v, fmaxf(-max_v, control[0]));
            float w = fminf(max_w, fmaxf(-max_w, control[1]));
            state[0] += __cosf(state[2]) * v * dt;
            state[1] += __sinf(state[2]) * v * dt;
            state[2] += w * dt;
        }

        __device__ bool is_collision(const float* state,
                                    const float* grid, int map_w, int map_h,
                                    float res, float ox, float oy) {
            int mx = (int)((state[0] - ox) / res);
            int my = (int)((state[1] - oy) / res);
            if (mx < 0 || mx >= map_w || my < 0 || my >= map_h) return true;
            int idx = my * map_w + mx;
            return grid[idx] > 50.0f;   // 与 CPU 阈值一致
        }

        __device__ float compute_cost(const float* state, const float* target, const float* weights,
                                    const float* control, const float* R,
                                    const float* grid, int map_w, int map_h,
                                    float res, float ox, float oy,
                                    float collision_cost) {
            float sc = 0.0f;
            #pragma unroll
            for (int i=0;i<3;i++){
                float d = state[i] - target[i];
                sc += weights[i]*d*d;
            }
            float uc = R[0]*control[0]*control[0] + R[1]*control[1]*control[1];

            if (is_collision(state, grid, map_w, map_h, res, ox, oy)) {
                sc += collision_cost;
            }
            return sc + uc;
        }

        // 把 U_d 放在 constant（很小），OK
        __device__ __constant__ float U_d[{{timesteps}}*2];

        extern "C"
        __global__ void rollout_kernel(
            float* costs, const float* noise,
            const float* init_state, const float* std_n, const float* R,
            const float* weights, const float* targets,
            const float* grid, int map_w, int map_h, float res, float ox, float oy,
            float dt, float max_v, float max_w, float collision_cost,
            int num_samples, int timesteps
        ){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_samples) return;

            float state[3];
            state[0] = init_state[0];
            state[1] = init_state[1];
            state[2] = init_state[2];

            float tot = 0.0f;
            int base = idx * timesteps * 2;

            for (int t=0; t<timesteps; ++t){
                float u0 = U_d[t*2+0] + std_n[0]*noise[base + t*2 + 0];
                float u1 = U_d[t*2+1] + std_n[1]*noise[base + t*2 + 1];
                float u[2] = {u0, u1};

                turtlebot3_kinematics(state, u, dt, max_v, max_w);
                tot += compute_cost(state, targets, weights, u, R,
                                    grid, map_w, map_h, res, ox, oy, collision_cost);
            }
            costs[idx] = tot;
        }
        """)
        return cuda_template.render(
            timesteps=self.num_timesteps
        )
