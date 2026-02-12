import logging
import time
import socket
import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)

from .config_xarm_vr_teleop import xArm7_VR_TeleopConfig
from .robot_observer import Robot_Observer

class xArm7_VR_Teleop(Teleoperator):
    config_class = xArm7_VR_TeleopConfig
    name = "xArm7 VR Teleoperator"

    def __init__(self, config: xArm7_VR_TeleopConfig):
        super().__init__(config)
        self.config = config
        # left controller socket
        self.sock_left = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_left.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_left.bind((self.config.local_host, int(self.config.port_left)))
        self.sock_left.setblocking(False)
        # right controller socket
        self.sock_right = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_right.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_right.bind((self.config.local_host, int(self.config.port_right)))
        self.sock_right.setblocking(False)
        # Robot vars
        self.robot_observer_right = Robot_Observer(self.config.robot_right)
        self.robot_observer_left = Robot_Observer(self.config.robot_left)


    @property
    def action_features(self) -> dict[str, type]:
        return {
            "right_joint_1.pos": float,
            "right_joint_2.pos": float,
            "right_joint_3.pos": float,
            "right_joint_4.pos": float,
            "right_joint_5.pos": float,
            "right_joint_6.pos": float,
            "right_joint_7.pos": float,
            "right_gripper.pos": float,
            "left_joint_1.pos": float,
            "left_joint_2.pos": float,
            "left_joint_3.pos": float,
            "left_joint_4.pos": float,
            "left_joint_5.pos": float,
            "left_joint_6.pos": float,
            "left_joint_7.pos": float,
            "left_gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return True    

    #@check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self) -> None:
        logger.info(f"{self} calibrated.")

    def configure(self) -> None:
        logger.info(f"{self} configured.")

    def setup_motors(self) -> None:
        logger.info(f"{self} motors setup.")

    #@check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()

        # Drain buffers to get latest data
        latest_data_bytes_left = None
        while True:
            try:
                data, _ = self.sock_left.recvfrom(4096)
                latest_data_bytes_left = data
            except BlockingIOError:
                break
        latest_data_bytes_right = None
        while True:
            try:
                data, _ = self.sock_right.recvfrom(4096)
                latest_data_bytes_right = data
            except BlockingIOError:
                break
        
        # process data and get action for both arms
        action_right = self.robot_observer_right.process_inputs(latest_data_bytes_right)    
        action_left = self.robot_observer_left.process_inputs(latest_data_bytes_left)

        # write action to dict
        action = np.hstack((action_right, action_left))
        action = {f"right_joint_{i+1}.pos": val for i, val in enumerate(action_right[:-1])}
        action["right_gripper.pos"] = action_right[-1]
        action.update({f"left_joint_{i+1}.pos": val for i, val in enumerate(action_left[:-1])})
        action["left_gripper.pos"] = action_left[-1]

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

    #@check_if_not_connected
    def disconnect(self) -> None:
        self.sock_left.close()
        self.sock_right.close()
        logger.info(f"{self} disconnected.")
        
