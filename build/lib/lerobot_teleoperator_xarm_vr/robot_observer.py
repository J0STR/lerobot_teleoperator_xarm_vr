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
                               1.630096197128296,
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
        self.last_pos = np.array([0.0, 0.0, 0.0])
        self.last_rot = R.from_quat([0.0, 0.0, 0.0, 1.0]) # identity quaternion
        self.max_rot_step = 0.2 #rad
        self.dt = 1/30 # 30 Hz
        self.v_joints = np.pi/4 # 90 deg/s
        self.v_xyz = 100 # mm/s

        self.button_already_pressed = False

    def process_inputs(self, latest_data_bytes):
        
        current_joints = self.read_joints()
        current_grip = self.gripper_pos
        action = np.hstack((current_joints, current_grip))

        if latest_data_bytes is not None:
            formated_data = process_controller_data(latest_data_bytes) 
        else:
            return action      
        # check buttons
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

        if formated_data["btn_ax"]:
            # Get current raw values from Godot
            curr_pos = np.array(formated_data["pos"])
            # Godot sends [x, y, z, w]. Scipy accepts [x, y, z, w]
            curr_quat = np.array(formated_data["quat"]) 

            if not self.button_already_pressed:
                # FIRST FRAME of press: Just store the state
                self.last_pos = curr_pos
                self.last_rot = R.from_quat(curr_quat)
                self.button_already_pressed = True
            
            else:
                # --- POSITION CALCULATION ---
                delta_pos = curr_pos - self.last_pos
                delta_pos_mm = delta_pos * 1000                    
                # Remap Position (Godot -> Robot)
                # Godot X -> Robot X (inverted)
                # Godot Y -> Robot Z
                # Godot Z -> Robot Y
                if formated_data["hand"] == "right":
                    remap_pos = np.array([-delta_pos_mm[0], delta_pos_mm[2], delta_pos_mm[1]])
                if formated_data["hand"] == "left":
                    remap_pos = np.array([delta_pos_mm[0], -delta_pos_mm[2], delta_pos_mm[1]])

                # --- ROTATION CALCULATION ---
                curr_rot_obj = R.from_quat(curr_quat)                    
                # Calculate the relative rotation: Diff = Current * Inverse(Last)
                delta_rot_obj = curr_rot_obj * self.last_rot.inv()                    
                # Convert to Rotation Vector (which xArm calls "Axis Angle Pose" rx/ry/rz)
                # This returns a 3D vector where length = angle, direction = axis
                rot_vec = delta_rot_obj.as_rotvec()
                # Remap Rotation Axis 
                # We apply the SAME coordinate shuffle to the rotation vector
                if formated_data["hand"] == "right":
                    remap_rot = np.array([-rot_vec[0], rot_vec[2], rot_vec[1]])
                if formated_data["hand"] == "left":
                    remap_rot = np.array([rot_vec[0], -rot_vec[2], rot_vec[1]])

                # --- SAFETY CLIPPING ---
                # Clip position speed (mm per step)
                v_limit = self.v_xyz * self.dt
                remap_pos = np.clip(remap_pos, -v_limit, v_limit)

                # Clip rotation speed (radians per step)
                # If we try to rotate too fast, the servo mode might error out
                norm = np.linalg.norm(remap_rot)
                if norm > self.v_joints*self.dt:
                    remap_rot = (remap_rot / norm) * self.v_joints*self.dt

                # Combine [x, y, z, rx, ry, rz]
                full_pose = np.hstack((remap_pos, remap_rot))

                # calculate target joints for VLA
                # --- 1. Read current robot state (position + orientation) ---
                current_absolute_aa = self.read_position()
                current_pos = np.array(current_absolute_aa[:3])
                current_rot_vec = np.array(current_absolute_aa[3:])
                # --- 2. Convert current orientation to a Scipy Rotation Object ---
                current_rot_obj = R.from_rotvec(current_rot_vec)
                # --- 3. Convert your DELTA (remap_rot) to a Scipy Rotation Object ---
                delta_rot_obj = R.from_rotvec(remap_rot)
                # --- 4. COMBINE rotations (Matrix Multiplication) ---
                target_rot_obj = delta_rot_obj * current_rot_obj
                # --- 5. Convert to RPY (Euler) for your solver ---
                target_rpy = target_rot_obj.as_euler('xyz', degrees=False) 
                # --- 6. Calculate Target Position ---
                target_pos = current_pos + remap_pos
                # --- 7. Pass to IK Solver ---
                target_pose_rpy = np.hstack((target_pos, target_rpy))
                code, angle = self.robot.arm.get_inverse_kinematics(target_pose_rpy, input_is_radian=True,return_is_radian=True)          

                # Update "Last" values for the next loop
                self.last_pos = curr_pos
                self.last_rot = curr_rot_obj
             
                
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