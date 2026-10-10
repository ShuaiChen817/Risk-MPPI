# Source this file in each terminal: source /path/to/Risk-MPPI/tools/env.sh
export RISK_MPPI_WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RISK_MPPI_PKG="$RISK_MPPI_WS/src/Risk-MPPI"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    printf '%s\n' 'ROS 2 Humble was not found at /opt/ros/humble.' >&2
    return 1
fi
# ROS 2 Humble uses Ubuntu system Python rather than an active Conda interpreter.
export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
if [[ ! -f "$RISK_MPPI_WS/install/setup.bash" ]]; then
    printf '%s\n' "Build this workspace first: $RISK_MPPI_WS" >&2
    return 1
fi
source "$RISK_MPPI_WS/install/setup.bash"

export TURTLEBOT3_MODEL=waffle_pi
export GAZEBO_MODEL_PATH="$RISK_MPPI_PKG/models:$RISK_MPPI_PKG/world${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
export GAZEBO_PLUGIN_PATH="$RISK_MPPI_WS/install/humans_actor_plugins/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
