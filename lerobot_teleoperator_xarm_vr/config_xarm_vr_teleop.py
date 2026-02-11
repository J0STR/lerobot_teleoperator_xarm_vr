from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("xArm7_Teleop")
@dataclass
class xArm7_VR_TeleopConfig(TeleoperatorConfig):
    # Your configuration fields go here
    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True