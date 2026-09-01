from xarm.wrapper import XArmAPI
import numpy as np
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

init_pose = np.array([-0.020695317536592484,
                       -0.979644238948822,
                         -0.008580705150961876,
                           0.6497616767883301,
                             0.018501725047826767,
                               1.630096197128296-(np.pi/18),
                                 -0.0007919175550341606])

def process_controller_data(data_bytes):
        try:
            msg = json.loads(data_bytes.decode('utf-8'))
            return msg
        except Exception as e:
            print(f"JSON Error: {e}")

class Robot_Observer():
    def __init__(self, ip: str, g2:bool=False):
        self.robot = XArmAPI(ip)
        self.init_pose = init_pose

        self.g2 = g2
        pos = self.read_gripper()
        self.gripper_pos = pos

        self.gripper_max = 840.0
        self.max_rot_step = 0.2 #rad
        self.dt = 1/30 # 30 Hz
        self.v_joints = 3*np.pi/4 # 90 deg/s
        self.v_xyz = 100 # mm/s

        ## vars for arm movement
        self.pos_when_triggered = None
        self.rot_when_triggered = None
        self.last_rot = None
        self.robot_pos_at_trigger = None
        self.robot_rot_at_trigger = None
        self.button_already_pressed = False
        self.rot_offset = None

        self.button_already_pressed = False

    def process_inputs(self, latest_data_bytes):
        
        current_joints = self.read_joints()
        current_grip = self.gripper_pos
        action = np.hstack((current_joints, current_grip))
        current_absolute_aa = self.read_position()
        current_pos = np.array(current_absolute_aa[:3])
        current_rot_vec = np.array(current_absolute_aa[3:])

        if latest_data_bytes is not None:
            formated_data = process_controller_data(latest_data_bytes) 
        else:
            return action      
        
        # close gripper
        if formated_data['trigger']:
            #self.gripper_pos = self.read_gripper()
            self.gripper_pos -= 20
            self.gripper_pos = np.clip(self.gripper_pos, 0, self.gripper_max)
            action[-1] = self.gripper_pos
        
        # open gripper
        if formated_data['grip']:
            #self.gripper_pos = self.read_gripper()
            self.gripper_pos += 20
            self.gripper_pos = np.clip(self.gripper_pos, 0, self.gripper_max)
            action[-1] = self.gripper_pos
        
        # reset button
        if formated_data["btn_by"]:
            code_robo, [error_code_robo, warn_code]= self.robot.get_err_warn_code()
            if code_robo == 0 and error_code_robo!=0: 
                self.robot.motion_enable(enable=True)
                self.robot.set_mode(1)
                self.robot.set_state(0)
            else:
                delta = self.init_pose - current_joints
                delta = np.clip(delta,-self.v_joints*self.dt,self.v_joints*self.dt)
                angles = current_joints + delta
                action[:-1] = angles

        # arm movement
        if formated_data["btn_ax"]:
            # get controller pos
            curr_pos = np.array(formated_data["pos"])
            curr_quat = np.array(formated_data["quat"]) 
            # calc rotation offset between robot and controller
            curr_vr_rot_raw = R.from_quat(curr_quat)
            vr_rot_vec = curr_vr_rot_raw.as_rotvec()
            if formated_data["hand"] == "right":
                remap_vr_curr_vec = np.array([
                        -vr_rot_vec[0], 
                        -vr_rot_vec[2], 
                        -vr_rot_vec[1]  
                    ])
            if formated_data["hand"] == "left":
                remap_vr_curr_vec = np.array([
                        vr_rot_vec[0],
                        vr_rot_vec[2], 
                        -vr_rot_vec[1] 
                    ])                
            remap_vr_curr_obj = R.from_rotvec(remap_vr_curr_vec) 

            # check if button pressed
            if not self.button_already_pressed:
                # safe controller and robot pos as reference
                self.pos_when_triggered = curr_pos
                self.robot_rot_at_trigger = R.from_rotvec(current_rot_vec)
                self.robot_pos_at_trigger = current_pos
                # save controller and robot offset to keep control axis
                # when the arm is rotated
                self.rot_offset = self.robot_rot_at_trigger * remap_vr_curr_obj.inv()   
                self.last_rot = R.from_quat(curr_quat)                 
                # button state
                self.button_already_pressed = True
            else:
                # calcuate movement based on the reference
                delta_pos = curr_pos - self.pos_when_triggered
                delta_pos_mm = delta_pos * 1000
                # map rotation to robot frame
                if formated_data["hand"] == "right":
                    remap_pos = np.array([-delta_pos_mm[0],
                                          delta_pos_mm[2],
                                          delta_pos_mm[1]])
                if formated_data["hand"] == "left":
                    remap_pos = np.array([delta_pos_mm[0],
                                          -delta_pos_mm[2],
                                          delta_pos_mm[1]])
                # map rotation to offset
                target_pos = self.robot_pos_at_trigger + remap_pos

                # ROTATION CALCULATION
                curr_rot_obj = R.from_quat(curr_quat)                    
                # Calculate the relative rotation: Diff = Current * Inverse(Last)
                delta_rot_obj = curr_rot_obj * self.last_rot.inv()                    
                rot_vec = delta_rot_obj.as_rotvec()
                # Remap Rotation Axis 
                # We apply the SAME coordinate shuffle to the rotation vector
                if formated_data["hand"] == "right":
                    remap_rot = np.array([-rot_vec[0], rot_vec[2], rot_vec[1]])
                if formated_data["hand"] == "left":
                    remap_rot = np.array([rot_vec[0], -rot_vec[2], rot_vec[1]])

                # Clip rotation speed (radians per step)
                norm = np.linalg.norm(remap_rot)
                if norm > self.v_joints*self.dt:
                    remap_rot = (remap_rot / norm) * self.v_joints*self.dt

                current_rot_vec = np.array(current_absolute_aa[3:])
                current_rot_obj = R.from_rotvec(current_rot_vec)
                delta_rot_obj = R.from_rotvec(remap_rot)
                target_rot_obj = delta_rot_obj * current_rot_obj
                target_rpy = target_rot_obj.as_euler('xyz', degrees=False) 

                target_pose_rpy = np.hstack((target_pos, target_rpy))
                code, angle = self.robot.arm.get_inverse_kinematics(target_pose_rpy, input_is_radian=True,return_is_radian=True)          
                # Update "Last" values for the next loop
                self.last_rot = curr_rot_obj

                if code == 0:
                    # Return action
                    action[:-1] = angle
        else:
            self.button_already_pressed = False

        return action


    def read_position(self)->list[float]:
        code, position = self.robot.get_position_aa(is_radian=True)
        return position
    
    def read_joints(self)->list[float]:
        code, [joints, velocity, effort] = self.robot.get_joint_states(is_radian=True)
        return joints
    
    def read_gripper(self)->float:
        if self.g2:
            code, position = self.robot.get_gripper_g2_position()
            position = position*10
            return position
        code, position = self.robot.get_gripper_position()
        return position
    
    def inverse_kinematic(self, position):
        code, angles = self.robot.get_inverse_kinematics(position, input_is_radian=True,return_is_radian=True)
        return angles
    
    def destroy(self):
        self.robot.disconnect()