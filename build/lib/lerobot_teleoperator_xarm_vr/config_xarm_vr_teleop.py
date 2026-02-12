from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xArm7_VR_Teleop")
@dataclass
class xArm7_VR_TeleopConfig(TeleoperatorConfig):
    # Your configuration fields go here
    # Port to connect to the arm
    port_left: str= "5003"
    port_right: str= "5002"
    local_host: str = "127.0.0.1"

    robot_right = "10.2.134.152"
    robot_left = "10.2.134.151"

    # Whether to use degrees for angles
    use_degrees: bool = True