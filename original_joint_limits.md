# Original joint limits

Backup of the original URDF joint limits from [agrobot.urdf.xacro](src/agrobot_description/urdf/agrobot.urdf.xacro) and MoveIt acceleration/scaling limits from [joint_limits.yaml](src/agrobot_moveit_config/config/joint_limits.yaml) before changing `arm_joint_1` to `10` rpm (`1.0472 rad/s`) for single-motor bench testing.

## URDF joint limits

| Joint | Type | Lower | Upper | Velocity | Effort | Unit |
| --- | --- | --- | --- | --- | --- | --- |
| `rail_joint` | `prismatic` | `0` | `1.565` | `0.5` | `100` | m, m/s, N |
| `arm_joint_1` | `revolute` | `-pi` | `pi` | `0.4` | `37` | rad, rad/s, N·m |
| `arm_joint_2` | `revolute` | `-pi/2` | `pi/2` | `1.96` | `144` | rad, rad/s, N·m |
| `arm_joint_3` | `revolute` | `-4.5` | `1.5` | `3.65` | `77` | rad, rad/s, N·m |
| `arm_joint_4` | `revolute` | `-pi` | `pi` | `2` | `25` | rad, rad/s, N·m |
| `arm_joint_5` | `revolute` | `-5` | `5` | `3.5` | `24` | rad, rad/s, N·m |
| `arm_joint_6` | `revolute` | `0` | `2*pi/3` | `3.5` | `24` | rad, rad/s, N·m |

## MoveIt scaling and acceleration limits

- `default_velocity_scaling_factor`: `0.1`
- `default_acceleration_scaling_factor`: `0.1`
- `arm_joint_1`–`arm_joint_6` `max_acceleration`: `0.5` rad/s²
- `rail_joint` `max_acceleration`: `0.1` m/s²
