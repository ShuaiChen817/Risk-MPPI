#!/usr/bin/env python3
"""
log-MPPI 控制器（ROS 2，方案B）
- 优先跟踪 Nav2 的 /plan（绿色线）；若没有 /plan，则用 RViz 的 /goal_pose 生成直线插点路径
- 允许“边转向边前进”：外层速度策略根据朝向误差连续调制 v、w
- 近目标时自动减速，达到后停止
- 发布采样轨迹 MarkerArray(/mppi_samples) 可视化
- 使用 /local_costmap/costmap 做碰撞惩罚 + 一步碰撞预判（避免“穿障”）
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped, Point   # === NEW === Point
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray     # 用于采样可视化
from builtin_interfaces.msg import Duration                # === NEW === lifetime

from mppi_controller import TurtleBot3MPPI


class LogMPPIControllerNode(Node):
    def __init__(self):
        super().__init__('log_mppi_controller')

        # —— 参数 ——
        self.declare_parameter('control_frequency', 30.0)
        self.declare_parameter('max_linear_velocity', 1.5) #1.5 2.5最好
        self.declare_parameter('max_angular_velocity', 3.0)
        self.declare_parameter('robot_radius', 0.26)

        self.declare_parameter('num_samples', 1024)
        self.declare_parameter('time_horizon', 1.1) #再大就会出错
        self.declare_parameter('exploration_variance', [0.35, 0.60])
        self.declare_parameter('lambda_temperature', 0.65)
        self.declare_parameter('num_iterations', 1)
        self.declare_parameter('use_log_normal', True)
        self.declare_parameter('log_normal_mu', 0.0)
        self.declare_parameter('log_normal_std', 0.3)


        # 代价/碰撞
        self.declare_parameter('state_weights', [1.2, 1.2, 0.24])
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('collision_cost', 3200.0)         # 提供给 mppi_controller（若实现）
        self.declare_parameter('use_local_costmap', True)        # === NEW ===
        self.declare_parameter('collision_threshold', 50)        # === NEW === >thr 视为障碍
        self.declare_parameter('pass_costmap_to_mppi', False)  # 不把代价图传进 mppi 内核，先避免越界

        # === RiskMap 接入参数 ===
        self.declare_parameter('use_risk_map', True)
        self.declare_parameter('risk_map_topic', '/risk_map')
        self.declare_parameter('risk_unknown_as', 40)            # -1 未知格映射为该值 (0..100)
        self.declare_parameter('risk_collision_threshold', 85)    # >thr 视为“极高风险”等同碰撞


        # 外层速度策略（连续可调）
        self.declare_parameter('speed_gain', 2.6)
        self.declare_parameter('turn_gain',  1.5)
        self.declare_parameter('min_forward_speed', 0.45)
        self.declare_parameter('heading_ok_threshold', 1.05)   # 弧度
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('near_goal_slowdown_dist', 0.45)
        self.declare_parameter('near_goal_min_speed', 0.12)

        # 采样可视化
        self.declare_parameter('publish_samples', True)
        self.declare_parameter('samples_max', 24)
        self.declare_parameter('sample_lifetime', 1.0)          # === NEW === 秒

        self.declare_parameter('min_turn_rate', 0.5)              # 角度误差明显时的最小角速度(rad/s)
        self.declare_parameter('min_forward_when_turning', 0.32)  # 角度很大时也给点前进速度


        # —— 运行时状态 ——
        self.current_pose = None            # geometry_msgs/Pose
        self.current_velocity = (0.0, 0.0)  # (vx, wz)
        self.global_path: Path = None       # /plan 或 插点路径
        self.target_reached = False
        self.has_odom = False

        # === NEW === 代价图缓存
        self.have_costmap = False
        self.costmap = None                 # ndarray[h, w], int16/8
        self.costmap_res = 0.05
        self.costmap_origin = (0.0, 0.0)
        self.costmap_frame = 'odom'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # —— 订阅 ——
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.create_subscription(Path, '/plan', self.path_callback, qos)
        self.costmap_topic = (self.get_parameter('risk_map_topic').value
                              if bool(self.get_parameter('use_risk_map').value)
                              else '/local_costmap/costmap')
        self.create_subscription(OccupancyGrid, self.costmap_topic, self.costmap_callback, qos)
        self.get_logger().info(f'地图来源: {self.costmap_topic}')
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos)

        # —— 发布 ——
        self.cmd_vel_pub     = self.create_publisher(Twist, '/cmd_vel', qos)
        self.status_pub      = self.create_publisher(Bool, '/mppi_status', qos)
        self.local_plan_pub  = self.create_publisher(Path, '/local_plan', qos)
        self.samples_pub     = self.create_publisher(MarkerArray, '/mppi_samples', 1)

        # 回退日志抑制用
        self._last_log_reason = None        # === NEW ===

        # —— 初始化 MPPI ——
        self.initialize_mppi()

        # —— 控制定时器 ——
        control_frequency = float(self.get_parameter('control_frequency').value)
        self.dt = 1.0 / control_frequency   # === NEW === 给预判用
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('log-MPPI Controller Node 启动成功')
        self.get_logger().info('优先使用 /plan；无 /plan 时，用 RViz 目标自动生成直线路径')

        # === NEW: 命令低通用的历史值 ===
        self._prev_cmd_v = 0.0
        self._prev_cmd_w = 0.0
        


    # ============================ 初始化 MPPI ============================
    def initialize_mppi(self):
        robot_params = {
            'max_linear_velocity': float(self.get_parameter('max_linear_velocity').value),
            'max_angular_velocity': float(self.get_parameter('max_angular_velocity').value),
            'robot_radius': float(self.get_parameter('robot_radius').value),
            'collision_cost': float(self.get_parameter('collision_cost').value)
        }

        if bool(self.get_parameter('use_log_normal').value):
            dist_type = 1
            self.get_logger().info('使用 log-MPPI (Normal + LogNormal)')
        else:
            dist_type = 0
            self.get_logger().info('使用标准 MPPI (Normal)')

        LogN_info = [
            dist_type,
            float(self.get_parameter('log_normal_mu').value),
            float(self.get_parameter('log_normal_std').value)
        ]

        self.mppi = TurtleBot3MPPI(
            state_dim=3,
            control_dim=2,
            num_samples=int(self.get_parameter('num_samples').value),
            time_horizon=float(self.get_parameter('time_horizon').value),
            control_freq=float(self.get_parameter('control_frequency').value),
            exploration_variance=np.array(self.get_parameter('exploration_variance').value, dtype=np.float32),
            robot_params=robot_params,
            LogN_info=LogN_info,
            num_optimization_iterations=int(self.get_parameter('num_iterations').value),
            lambda_=float(self.get_parameter('lambda_temperature').value)
        )

    # ============================ 回调：里程计 ============================
    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        self.current_velocity = (float(msg.twist.twist.linear.x),
                                 float(msg.twist.twist.angular.z))
        
        # === 新增：从四元数计算 yaw（弧度） ===
        q = msg.pose.pose.orientation
        # yaw about Z
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.has_odom = True

    # ============================ 回调：/plan ============================
    def path_callback(self, msg: Path):
        if len(msg.poses) == 0:
            return
        self.global_path = msg
        self.target_reached = False
        self.get_logger().info(f'收到 /plan，{len(msg.poses)} 点')

    # ============================ 回调：局部代价图 ============================
    def costmap_callback(self, msg: OccupancyGrid):
        try:
            w, h = msg.info.width, msg.info.height
            data = np.array(msg.data, dtype=np.int16).reshape(h, w)   # [rows=h, cols=w]
            if bool(self.get_parameter('use_risk_map').value):
                unk = int(self.get_parameter('risk_unknown_as').value)
                data = np.where(data < 0, unk, data).astype(np.int16)

            self.costmap = data
            self.costmap_res = float(msg.info.resolution)
            self.costmap_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            self.costmap_frame = msg.header.frame_id or 'odom'
            self.have_costmap = True
            mode = 'riskmap' if bool(self.get_parameter('use_risk_map').value) else 'costmap'
            thr  = int(self.get_parameter('risk_collision_threshold').value)
            try:
                self.mppi.set_map_semantics(mode, thr)
            except Exception:
                pass

            self.mppi.update_costmap(
            data.ravel(),   # 传一维也行，MPPI内部会 reshape(h,w)
            {
                'width': w,
                'height': h,
                'resolution': self.costmap_res,
                'origin_x': self.costmap_origin[0],
                'origin_y': self.costmap_origin[1],
            }
        )
        except Exception as e:
            self.get_logger().error(f'处理代价地图失败: {e}')

    # ============================ 回调：RViz 目标 -> 直线插点 ============================
    def goal_callback(self, msg: PoseStamped):
        if not self.has_odom or self.current_pose is None:
            self.get_logger().warn('尚未收到 /odom，忽略 RViz 目标')
            return

        path = Path()
        path.header.frame_id = self.costmap_frame if self.have_costmap else 'odom'  # === NEW ===
        path.header.stamp = self.get_clock().now().to_msg()

        sx = float(self.current_pose.position.x)
        sy = float(self.current_pose.position.y)
        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)

        # 起点
        p0 = PoseStamped(); p0.header = path.header; p0.pose = self.current_pose
        path.poses.append(p0)

        # 直线插点（0.25m 间距）
        dist = float(np.hypot(gx - sx, gy - sy))
        n_mid = max(1, int(dist / 0.25))
        for i in range(1, n_mid):
            r = float(i) / float(n_mid)
            ip = PoseStamped(); ip.header = path.header
            ip.pose.position.x = sx + r * (gx - sx)
            ip.pose.position.y = sy + r * (gy - sy)
            ip.pose.position.z = 0.0
            ip.pose.orientation.w = 1.0
            path.poses.append(ip)

        # 终点
        pe = PoseStamped(); pe.header = path.header
        pe.pose.position.x = gx; pe.pose.position.y = gy; pe.pose.orientation.w = 1.0
        path.poses.append(pe)

        self.global_path = path
        self.target_reached = False
        self.get_logger().info(f'收到 RViz 目标: ({gx:.2f}, {gy:.2f})，生成直线插点路径 {len(path.poses)} 点')

    # ============================ 工具：当前状态 (x,y,yaw) ============================
    def get_current_state(self):
        if self.current_pose is None:
            return None
        q = self.current_pose.orientation
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([float(self.current_pose.position.x),
                         float(self.current_pose.position.y),
                         float(yaw)], dtype=np.float32)

    # ============================ 工具：从路径取“前瞻目标点” ============================
    def get_target_state(self, current_state):
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None

        lookahead = float(self.get_parameter('lookahead_distance').value)
        cx, cy = current_state[0], current_state[1]

        # 最近点
        closest_idx, min_d = 0, float('inf')
        for i, ps in enumerate(self.global_path.poses):
            dx = ps.pose.position.x - cx
            dy = ps.pose.position.y - cy
            d2 = dx*dx + dy*dy
            if d2 < min_d:
                min_d = d2
                closest_idx = i

        # 向前找到 >= lookahead 的点
        target_idx = len(self.global_path.poses) - 1
        for i in range(closest_idx, len(self.global_path.poses)):
            dx = self.global_path.poses[i].pose.position.x - cx
            dy = self.global_path.poses[i].pose.position.y - cy
            if math.hypot(dx, dy) >= lookahead:
                target_idx = i
                break

        # 目标 yaw 用路径切线
        if target_idx > 0:
            px = self.global_path.poses[target_idx - 1].pose.position.x
            py = self.global_path.poses[target_idx - 1].pose.position.y
        else:
            px = self.global_path.poses[target_idx].pose.position.x
            py = self.global_path.poses[target_idx].pose.position.y
        tx = self.global_path.poses[target_idx].pose.position.x
        ty = self.global_path.poses[target_idx].pose.position.y
        tyaw = math.atan2(ty - py, tx - px)

        return np.array([float(tx), float(ty), float(tyaw)], dtype=np.float32)

    # ============================ 备用控制（连续策略） ============================
    def simple_backup_control(self, current_state, target_state):
        dx = target_state[0] - current_state[0]
        dy = target_state[1] - current_state[1]
        dist = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        angle_err = (angle_to_target - current_state[2] + math.pi) % (2*math.pi) - math.pi

        # v 连续衰减；w 正比误差
        vmax = float(self.get_parameter('max_linear_velocity').value)
        v = np.clip(0.6 * dist * math.cos(min(abs(angle_err), math.pi/2)), 0.0, vmax)
        w = 1.2 * angle_err

        # 小误差给最小速度
        if abs(angle_err) < float(self.get_parameter('heading_ok_threshold').value):
            v = max(v, float(self.get_parameter('min_forward_speed').value))

        wmax = float(self.get_parameter('max_angular_velocity').value)
        w = float(np.clip(w, -wmax, wmax))
        return np.array([float(v), float(w)], dtype=np.float32)

    # ============================ 可视化：局部两点线段（当前->target） ============================
    def publish_local_plan(self, current_state, target_state):
        path = Path()
        path.header.frame_id = self.costmap_frame if self.have_costmap else 'odom'
        path.header.stamp = self.get_clock().now().to_msg()

        best = None
        if hasattr(self.mppi, 'get_best_trajectory'):
            try:
                best = self.mppi.get_best_trajectory()
            except Exception:
                best = None

        poses = []
        if best is not None and len(best) > 1:
            # 用最优轨迹
            for s in best:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(s[0])
                ps.pose.position.y = float(s[1])
                ps.pose.orientation.w = 1.0
                poses.append(ps)
        else:
            # 兜底：两点直线
            a = PoseStamped(); a.header = path.header
            a.pose.position.x = float(current_state[0]); a.pose.position.y = float(current_state[1]); a.pose.orientation.w = 1.0
            b = PoseStamped(); b.header = path.header
            b.pose.position.x = float(target_state[0]); b.pose.position.y = float(target_state[1]); b.pose.orientation.w = 1.0
            poses = [a, b]

        path.poses = poses
        self.local_plan_pub.publish(path)


    # ============================ 可视化：采样轨迹 ============================
    def publish_samples(self, current_state):
        if not bool(self.get_parameter('publish_samples').value):
            return

        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        frame_id = self.costmap_frame if self.have_costmap else 'odom'

        # 从 MPPI 取样本
        samples = None
        costs = None
        best_idx = -1

        if hasattr(self.mppi, 'get_debug_samples'):
            try:
                samples = self.mppi.get_debug_samples(max_count=int(self.get_parameter('samples_max').value))
            except Exception:
                samples = None

        if hasattr(self.mppi, 'get_debug_costs_and_best'):
            try:
                costs, best_idx = self.mppi.get_debug_costs_and_best()
            except Exception:
                costs, best_idx = None, -1

        # 线宽 & 寿命（更细更好看）
        line_width = 0.004
        lifetime_sec = float(self.get_parameter('sample_lifetime').value) if self.has_parameter('sample_lifetime') else 0.8

        # 末端外推长度（米）——只影响显示
        extend_len = 1.0

        def color_from_cost(i, cmin, cmax, best_i):
            # 最优绿色
            if i == best_i:
                return (0.1, 1.0, 0.1, 0.95)
            # 其它：红->黄渐变（差=红，稍差=橙黄）
            if cmax <= cmin or costs is None:
                return (1.0, 0.3, 0.3, 0.75)
            alpha = float(min(max((costs[i] - cmin) / (cmax - cmin), 0.0), 1.0))
            # 红(1,0,0) 到 黄(1,1,0) 插值
            r = 1.0
            g = 0.2 + 0.8 * (1.0 - alpha)
            b = 0.0
            a = 0.75
            return (r, g, b, a)

        if samples is not None and len(samples) > 0:
            K = min(len(samples), int(self.get_parameter('samples_max').value))
            cmin, cmax = (np.min(costs), np.max(costs)) if isinstance(costs, np.ndarray) else (0.0, 1.0)

            for i in range(K):
                pts = np.asarray(samples[i], dtype=np.float32)

                # —— 外推：用最后两点方向延长一小段（只为看起来更长）——
                if pts.shape[0] >= 2 and extend_len > 1e-3:
                    p_last = pts[-1, :2]
                    p_prev = pts[-2, :2]
                    d = p_last - p_prev
                    seg = float(np.hypot(d[0], d[1]))
                    if seg > 1e-4:
                        ux, uy = d / seg
                        # 按 0.05m 一段追加若干点
                        n_ext = int(extend_len / 0.05)
                        ext_pts = [p_last + (j+1)*0.05*np.array([ux, uy], dtype=np.float32) for j in range(n_ext)]
                        ext_pts = np.stack(ext_pts, axis=0)
                        # 拼起来（z=0.01）
                        ext_traj = np.zeros((pts.shape[0] + ext_pts.shape[0], 3), dtype=np.float32)
                        ext_traj[:pts.shape[0], :] = pts
                        ext_traj[pts.shape[0]:, 0:2] = ext_pts
                        ext_traj[pts.shape[0]:, 2] = pts[-1, 2]
                        pts = ext_traj

                mk = Marker()
                mk.header.frame_id = frame_id; mk.header.stamp = now
                mk.ns = 'mppi_samples'; mk.id = i
                mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
                mk.scale.x = line_width
                rr, gg, bb, aa = color_from_cost(i, cmin, cmax, best_idx)
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = rr, gg, bb, aa
                mk.pose.orientation.w = 1.0
                mk.lifetime = Duration(sec=int(lifetime_sec), nanosec=int((lifetime_sec % 1.0) * 1e9))
                mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.01) for p in pts]
                arr.markers.append(mk)
        # else:
        #     # 没有真实采样就画扇形占位（线也更细）
        #     N = int(self.get_parameter('samples_max').value)
        #     for i in range(N):
        #         ang = current_state[2] + (i - N//2) * (math.pi/24.0)
        #         L = 0.6
        #         x0, y0 = float(current_state[0]), float(current_state[1])
        #         x1, y1 = x0 + L*math.cos(ang), y0 + L*math.sin(ang)
        #         mk = Marker(); mk.header.frame_id = frame_id; mk.header.stamp = now
        #         mk.ns = 'mppi_samples_dummy'; mk.id = i
        #         mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
        #         mk.scale.x = 0.01
        #         mk.color.a = 0.25
        #         mk.color.g = 1.0
        #         mk.pose.orientation.w = 1.0
        #         mk.lifetime = Duration(sec=int(lifetime_sec), nanosec=int((lifetime_sec % 1.0) * 1e9))
        #         mk.points = [Point(x=x0,y=y0,z=0.01), Point(x=x1,y=y1,z=0.01)]
        #         arr.markers.append(mk)

        if samples is None:
            self.get_logger().warn("MPPI debug_samples is None → show dummy rays")

        self.samples_pub.publish(arr)


    # ============================ 栅格工具（世界→栅格 & 占用判定） ============================
    def _world_to_grid(self, x, y):
        ox, oy = self.costmap_origin
        col = int((x - ox) / self.costmap_res)
        row = int((y - oy) / self.costmap_res)
        return row, col

    def _is_occupied(self, x, y, thr=None):
        thr = self.get_parameter('collision_threshold').value if thr is None else thr
        if not self.have_costmap or self.costmap is None:
            return False
        r, c = self._world_to_grid(x, y)
        if 0 <= r < self.costmap.shape[0] and 0 <= c < self.costmap.shape[1]:
            return int(self.costmap[r, c]) > int(thr)
        return False
    
    def _costmap_gradient(self, x, y, rad_cells=2):
        """在 (x,y) 周围窗口做简单的“成本重心”估算，返回梯度 (gx, gy)（世界坐标方向）"""
        if not self.have_costmap or self.costmap is None:
            return 0.0, 0.0
        r0, c0 = self._world_to_grid(x, y)
        H, W = self.costmap.shape
        res = self.costmap_res
        gx = gy = 0.0
        total = 0.0
        for dr in range(-rad_cells, rad_cells + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H: 
                continue
            for dc in range(-rad_cells, rad_cells + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                val = int(self.costmap[rr, cc])
                if val <= 0:
                    continue
                total += val
                # 栅格中心坐标（世界系）
                wx = self.costmap_origin[0] + (cc + 0.5) * res
                wy = self.costmap_origin[1] + (rr + 0.5) * res
                # 指向成本重心的向量累加（近似梯度）
                gx += val * (wx - x)
                gy += val * (wy - y)
        if total > 1e-6:
            gx /= total; gy /= total
        return gx, gy
    def _obstacle_slowdown_factor(self, x, y, radius_m=0.45, thr=None):
        """
        统计 (x,y) 周围一个圆盘内的占用比例，返回 [0.3, 1.0] 的减速系数：
        - 占用多 -> 系数小（更慢）
        - 占用少 -> 接近 1
        """
        if not self.have_costmap or self.costmap is None:
            return 1.0
        thr = self.get_parameter('collision_threshold').value if thr is None else thr
        res = self.costmap_res
        r0, c0 = self._world_to_grid(x, y)
        H, W = self.costmap.shape
        rad_cells = max(1, int(radius_m / res))

        occ = 0
        tot = 0
        for dr in range(-rad_cells, rad_cells + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in range(-rad_cells, rad_cells + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                if dr*dr + dc*dc <= rad_cells*rad_cells:
                    tot += 1
                    if int(self.costmap[rr, cc]) > int(thr):
                        occ += 1
        if tot == 0:
            return 1.0
        ratio = occ / float(tot)  # 0~1
        # 线性映射到 [0.3, 1.0]（占比高 -> 0.3，低 -> 1.0）
        return max(0.3, 1.0 - 0.7 * ratio)



    # ============================ 主控制循环 ============================
    def control_loop(self):
        if not self.has_odom:
            return

        current_state = self.get_current_state()
        if current_state is None:
            self.publish_zero_velocity()
            return
        if self.target_reached:
            self.publish_zero_velocity()
            return

        target_state = self.get_target_state(current_state)
        if target_state is None:
            # 没有路径就停住（等待 /plan 或 RViz 目标）
            self.publish_zero_velocity()
            return

        # 到达检测
        goal_tol = float(self.get_parameter('goal_tolerance').value)
        if self.global_path and len(self.global_path.poses) > 0:
            goal = self.global_path.poses[-1].pose.position
            goal_dist = math.hypot(current_state[0] - goal.x, current_state[1] - goal.y)
            if goal_dist < goal_tol:
                if not self.target_reached:
                    self.get_logger().info('到达目标')
                    self.target_reached = True
                    self.status_pub.publish(Bool(data=True))
                self.publish_zero_velocity()
                return

        # 近目标减速
        near_dist = float(self.get_parameter('near_goal_slowdown_dist').value)
        near_min_v = float(self.get_parameter('near_goal_min_speed').value)
        slow_ratio = 1.0
        if self.global_path and len(self.global_path.poses) > 0:
            goal = self.global_path.poses[-1].pose.position
            d = math.hypot(current_state[0] - goal.x, current_state[1] - goal.y)
            if d < near_dist:
                slow_ratio = max(0.25, d / near_dist)

        weights = np.array(self.get_parameter('state_weights').value, dtype=np.float32)
        try:
            t0 = time.time()
            control, normalizer, min_cost = self.mppi.compute_control(current_state, target_state, weights)
            dtc = time.time() - t0

            # 仅当控制为 None/NaN 时回退（放宽回退条件）
            invalid = (control is None or np.any(np.isnan(control)))
            use_fallback = False
            reason = ""
            if invalid:
                use_fallback = True
                reason = "mppi_none_nan"

            if use_fallback:
                if self._last_log_reason != reason:
                    self.get_logger().warn(f'使用备用控制器，原因: {reason}')
                    self._last_log_reason = reason
                control = self.simple_backup_control(current_state, target_state)

            # === NEW === 一步碰撞预判（防穿障）：若前进一步会撞，则减 v、加 |w|
            # —— 两步预判 + 基于代价图梯度的转向偏置（更强的防撞兜底）——
            impending_hit = False
            if bool(self.get_parameter('use_local_costmap').value) and control is not None and self.have_costmap:
                v_try = float(control[0])
                w_try = float(control[1])

                x, y, yaw = float(current_state[0]), float(current_state[1]), float(current_state[2])

                def step(x, y, yaw, v, w, dt):
                    return (x + v * math.cos(yaw) * dt,
                            y + v * math.sin(yaw) * dt,
                            yaw + w * dt)

                # 一步、两步位置
                nx1, ny1, nyaw1 = step(x, y, yaw, v_try, w_try, self.dt)
                nx2, ny2, nyaw2 = step(nx1, ny1, nyaw1, v_try, w_try, self.dt)

                occ1 = self._is_occupied(nx1, ny1)
                occ2 = self._is_occupied(nx2, ny2)
                impending_hit = (occ1 or occ2)

                # 拥挤区整体降速（0.3~1.0）
                # crowd_k = self._obstacle_slowdown_factor(x, y, radius_m=0.45)
                # v_try *= crowd_k

                if impending_hit:
                  
                    # 让 MPPI 在代价里处理避障（已启用近障惩罚），外层不再强行改 w
                    self.get_logger().warn('预判将碰撞：交由 MPPI 代价处理（近障惩罚已开启）')
                    # 如果你特别担心穿障，可以做 VERY 轻的减速：
                    # v_try = max(v_try, float(self.get_parameter('min_forward_when_turning').value))
                    control = np.array([v_try, w_try], dtype=np.float32)

                    


            # 发布控制
            self.publish_control(control, current_state, target_state, slow_ratio, near_min_v)

            # 可视化
            self.publish_local_plan(current_state, target_state)
            self.publish_samples(current_state)

            if dtc > 0.02:
                self.get_logger().warn(f'控制计算耗时: {dtc*1000:.1f}ms')
        except Exception as e:
            self.get_logger().error(f'MPPI控制异常: {e}，使用备用控制器')
            control = self.simple_backup_control(current_state, target_state)
            self.publish_control(control, current_state, target_state, slow_ratio, near_min_v)

    # ============================ 发布速度（带外层增益 & 连续角度逻辑） ============================
    def publish_control(self, control, current_state, target_state, slow_ratio=1.0, near_min_v=0.0):
        # 基于 MPPI 的控制量
        v = float(control[0]) * float(self.get_parameter('speed_gain').value)
        w = float(control[1]) * float(self.get_parameter('turn_gain').value)

        # —— 角度误差 —— 
        dx = target_state[0] - current_state[0]
        dy = target_state[1] - current_state[1]
        angle_to_target = math.atan2(dy, dx)
        ang_err = (angle_to_target - current_state[2] + math.pi) % (2*math.pi) - math.pi

        # —— 角度越大，线速越低；给地板值避免完全停住 —— 
        k = 0.65 + 0.35 * math.cos(min(abs(ang_err), math.pi/2))
        v *= k

        # 角度大也给一点前进速度，防止死等原地转
        v = max(v, float(self.get_parameter('min_forward_when_turning').value))

        if abs(ang_err) < float(self.get_parameter('heading_ok_threshold').value):
            v = max(v, float(self.get_parameter('min_forward_speed').value))

        # —— 按“曲率”压速度：|w| 越大 v 上限越低（避免高速猛拐）——
        # 经验：v_max_curv = Vmax / (1 + 0.9*|w|) 你可以把 0.9 再调
        vmax_phys = float(self.get_parameter('max_linear_velocity').value)
        v_max_curv = vmax_phys / (1.0 + 0.6 * abs(w))
        v = min(v, v_max_curv)

        # —— 近目标减速 —— 
        v *= slow_ratio
        if slow_ratio < 0.5:
            v = max(v, near_min_v)

        # —— 拥挤区整体降速（0.3~1.0） —— 
        # crowd_k = self._obstacle_slowdown_factor(current_state[0], current_state[1], radius_m=0.45)
        # v *= crowd_k

        # 不倒车（如需允许倒车，删掉这段）
        if v < 0.0:
            v = 0.0

        # 限幅
        wmax = float(self.get_parameter('max_angular_velocity').value)
        v = float(np.clip(v, 0.0, vmax_phys))
        w = float(np.clip(w, -wmax, wmax))

        # —— 输出低通（alpha 0.5 可调大些更稳，但更“迟钝”）——
        alpha = 0.5
        v = alpha * v + (1 - alpha) * self._prev_cmd_v
        w = alpha * w + (1 - alpha) * self._prev_cmd_w
        self._prev_cmd_v, self._prev_cmd_w = v, w

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_vel_pub.publish(cmd)


    def publish_zero_velocity(self):
        self.cmd_vel_pub.publish(Twist())

    def path_callback(self, msg: Path):
        if len(msg.poses) == 0:
            return
        # ...（帧检查略）
        self.global_path = msg
        self.target_reached = False

        # ★ 新 plan 到来 → 重置 MPPI 内部名义控制 & 轨迹
        if hasattr(self, 'mppi') and self.mppi is not None:
            self.mppi.reset_nominal()     # 需要在内核里实现（见下）
            self.mppi.clear_best_traj()



def main(args=None):
    rclpy.init(args=args)
    try:
        node = LogMPPIControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'节点错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
