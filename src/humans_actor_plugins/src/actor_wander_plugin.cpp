#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/common/Console.hh>
#include <gazebo/common/Plugin.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>
#include <string>
#include <cctype>

namespace gazebo {

class ActorWanderPlugin : public gazebo::ModelPlugin {
public:
  ActorWanderPlugin() : rng_(std::random_device{}()) {}

  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
  {
    actor_ = boost::dynamic_pointer_cast<physics::Actor>(_model);
    world_ = actor_->GetWorld();
    lastSimTime_ = world_->SimTime();

    // 可选参数（不要求都有）
    if (_sdf->HasElement("xmin")) xmin_ = _sdf->Get<double>("xmin");
    if (_sdf->HasElement("xmax")) xmax_ = _sdf->Get<double>("xmax");
    if (_sdf->HasElement("ymin")) ymin_ = _sdf->Get<double>("ymin");
    if (_sdf->HasElement("ymax")) ymax_ = _sdf->Get<double>("ymax");
    if (_sdf->HasElement("speed_min")) vmin_ = _sdf->Get<double>("speed_min");
    if (_sdf->HasElement("speed_max")) vmax_ = _sdf->Get<double>("speed_max");
    if (_sdf->HasElement("animation_factor")) animFactor_ = _sdf->Get<double>("animation_factor");
    if (_sdf->HasElement("animate")) animate_ = _sdf->Get<bool>("animate");
    if (_sdf->HasElement("animation_period")) animPeriod_ = _sdf->Get<double>("animation_period");

    // 运动模式：wander 或 ellipse
    if (_sdf->HasElement("mode")) mode_ = _sdf->Get<std::string>("mode");
    std::string mlc = mode_;
    std::transform(mlc.begin(), mlc.end(), mlc.begin(), [](unsigned char c){ return std::tolower(c); });
    use_ellipse_ = (mlc == "ellipse");
    // 初始随机散布
    if (_sdf->HasElement("random_spawn")) random_spawn_ = _sdf->Get<bool>("random_spawn");
    // 椭圆参数
    if (_sdf->HasElement("a")) a_axis_ = _sdf->Get<double>("a");
    if (_sdf->HasElement("b")) b_axis_ = _sdf->Get<double>("b");
    if (_sdf->HasElement("omega")) omega_ = _sdf->Get<double>("omega");
    if (_sdf->HasElement("angular_speed")) omega_ = _sdf->Get<double>("angular_speed");
    if (_sdf->HasElement("period")) { double T = _sdf->Get<double>("period"); if (T > 1e-6) omega_ = 2.0*M_PI / T; }
    if (_sdf->HasElement("phase")) theta_ = _sdf->Get<double>("phase");
    if (_sdf->HasElement("clockwise")) clockwise_ = _sdf->Get<bool>("clockwise");
    if (_sdf->HasElement("face_forward")) face_forward_ = _sdf->Get<bool>("face_forward");
    if (_sdf->HasElement("cx")) cx_ = _sdf->Get<double>("cx");
    if (_sdf->HasElement("cy")) cy_ = _sdf->Get<double>("cy");

    // 重置脚本时间到 0，避免加载包含大 sim_time 的世界状态时越界
    if (actor_) {
      try { actor_->SetScriptTime(0.0); } catch (...) {}
    }

    gzmsg << "[ActorWanderPlugin] Loaded for model '" << _model->GetName()
          << "' | animate=" << (animate_ ? "true" : "false")
          << ", anim_factor=" << animFactor_
          << ", anim_period=" << animPeriod_ << std::endl;

    // 初始化
    std::uniform_real_distribution<double> dang(-M_PI, M_PI);
    std::uniform_real_distribution<double> dspd(vmin_, vmax_);
    yaw_ = dang(rng_);
    speed_ = dspd(rng_);

    // 记录初始 z
    init_z_ = actor_->WorldPose().Pos().Z();

    // 初始随机散布
    if (random_spawn_) {
      std::uniform_real_distribution<double> dx(xmin_, xmax_);
      std::uniform_real_distribution<double> dy(ymin_, ymax_);
      ignition::math::Pose3d p = actor_->WorldPose();
      p.Pos().X() = dx(rng_);
      p.Pos().Y() = dy(rng_);
      p.Pos().Z() = init_z_;
      p.Rot() = ignition::math::Quaterniond(0,0,yaw_);
      actor_->SetWorldPose(p, true, true);
    }

    // 椭圆中心默认取初始位置
    if (use_ellipse_) {
      ignition::math::Pose3d p0 = actor_->WorldPose();
      if (!_sdf->HasElement("cx")) cx_ = p0.Pos().X();
      if (!_sdf->HasElement("cy")) cy_ = p0.Pos().Y();
      omega_ = (clockwise_ ? -std::abs(omega_) : std::abs(omega_));
    }

    // 在 WorldUpdateEnd 阶段更新位姿，避免被 Actor 的脚本更新覆盖
    // WorldUpdateEnd 的回调签名为 void()，不带 UpdateInfo
    updateConn_ = event::Events::ConnectWorldUpdateEnd(
      std::bind(&ActorWanderPlugin::OnUpdate, this));
  }

  void OnUpdate()
  {
    if (!world_ || !actor_) return;
    // 通过 world_->SimTime() 自己计算 dt
    common::Time now = world_->SimTime();
    double dt = (now - lastSimTime_).Double();
    if (dt <= 0) return;
    lastSimTime_ = now;

    ignition::math::Pose3d pose = actor_->WorldPose();

    if (use_ellipse_) {
      // 椭圆轨迹
      theta_ += omega_ * dt;
      double x = cx_ + a_axis_ * std::cos(theta_);
      double y = cy_ + b_axis_ * std::sin(theta_);
      double yaw = yaw_;
      if (face_forward_) {
        double dx = -a_axis_ * std::sin(theta_);
        double dy =  b_axis_ * std::cos(theta_);
        yaw = std::atan2(dy, dx);
      }
      pose.Pos().X() = x;
      pose.Pos().Y() = y;
      pose.Pos().Z() = init_z_;
      pose.Rot() = ignition::math::Quaterniond(0,0,yaw);
      actor_->SetWorldPose(pose, true, true);
    } else {
      // 随机游走
      ignition::math::Vector3d pos = pose.Pos();
      ignition::math::Vector3d f(std::cos(yaw_), std::sin(yaw_), 0);
      ignition::math::Vector3d next = pos + f * (speed_ * dt);

      bool bounced = false;
      if (next.X() < xmin_ || next.X() > xmax_) { yaw_ = M_PI - yaw_; bounced = true; }
      if (next.Y() < ymin_ || next.Y() > ymax_) { yaw_ = -yaw_;       bounced = true; }
      if (bounced) {
        while (yaw_ >  M_PI) yaw_ -= 2*M_PI;
        while (yaw_ < -M_PI) yaw_ += 2*M_PI;
        f.Set(std::cos(yaw_), std::sin(yaw_), 0);
        next = pos + f * (speed_ * dt);
      }
      pose.Set(next, ignition::math::Quaterniond(0,0,yaw_));
      actor_->SetWorldPose(pose, true, true);
    }

    // 推进动画（仅当显式开启）
    if (animate_ && animFactor_ != 0.0) {
      try {
        double t = actor_->ScriptTime() + animFactor_ * dt;
        if (animPeriod_ > 1e-6) {
          // 循环到 [0, animPeriod_)
          t = std::fmod(std::fmod(t, animPeriod_) + animPeriod_, animPeriod_);
        }
        actor_->SetScriptTime(t);
      } catch (...) {
        // 无脚本/无轨迹时静默忽略
      }
    }
  }

  void Reset() override
  {
    gazebo::ModelPlugin::Reset();
    if (world_) lastSimTime_ = world_->SimTime();
    yaw_ = 0.0;
    speed_ = std::max(0.3, vmin_);
  }

private:
  physics::WorldPtr world_;
  physics::ActorPtr actor_;
  event::ConnectionPtr updateConn_;

  double xmin_=-10, xmax_=10, ymin_=-10, ymax_=10;
  double vmin_=0.3, vmax_=0.9;
  double animFactor_=1.0;
  bool animate_ = false;     // 是否推进骨骼动画时间（缺省关闭）
  double animPeriod_ = 1.0;  // 动画脚本周期（秒），用于时间取模
  std::string mode_ = "wander";
  bool use_ellipse_ = false;
  bool random_spawn_ = true; // 缺省在边界内散布
  // 椭圆参数
  double cx_ = 0.0, cy_ = 0.0;      // 中心
  double a_axis_ = 4.0, b_axis_ = 2.5; // 半轴
  double omega_ = 0.25;              // 角速度(rad/s)
  double theta_ = 0.0;               // 当前相位
  bool clockwise_ = false;
  bool face_forward_ = true;

  double yaw_=0.0, speed_=0.5;
  common::Time lastSimTime_;
  std::mt19937 rng_;
  double init_z_ = 1.0;
};

} // namespace gazebo

// Register as a ModelPlugin (ActorPlugin derives from ModelPlugin)
GZ_REGISTER_MODEL_PLUGIN(gazebo::ActorWanderPlugin)
