# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#modified by Logan
"""Deploy a PyTorch policy to C MuJoCo and play with it."""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
# from keyboard_reader import KeyboardController

POLICY_PATH = "policy.pt"


class G1MjxJointIndex:
    """Joint indices based on the order in g1_mjx_alt.xml (23 DoF model)."""
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    LeftShoulderPitch = 13
    LeftShoulderRoll = 14
    LeftShoulderYaw = 15
    LeftElbow = 16
    LeftWristRoll = 17
    RightShoulderPitch = 18
    RightShoulderRoll = 19
    RightShoulderYaw = 20
    RightElbow = 21
    RightWristRoll = 22


class G1PyTorchJointIndex:
    """Joint indices based on the order in your PyTorch model."""
    # Actual joint order from PyTorch training:
    # ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
    # 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    # 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
    # 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
    # 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']
    
    LeftHipPitch = 0       # left_hip_pitch_joint
    RightHipPitch = 1      # right_hip_pitch_joint
    WaistYaw = 2           # waist_yaw_joint
    LeftHipRoll = 3        # left_hip_roll_joint
    RightHipRoll = 4       # right_hip_roll_joint
    LeftHipYaw = 5         # left_hip_yaw_joint
    RightHipYaw = 6        # right_hip_yaw_joint
    LeftKnee = 7           # left_knee_joint
    RightKnee = 8          # right_knee_joint
    LeftShoulderPitch = 9  # left_shoulder_pitch_joint
    RightShoulderPitch = 10 # right_shoulder_pitch_joint
    LeftAnklePitch = 11    # left_ankle_pitch_joint
    RightAnklePitch = 12   # right_ankle_pitch_joint
    LeftShoulderRoll = 13  # left_shoulder_roll_joint
    RightShoulderRoll = 14 # right_shoulder_roll_joint
    LeftAnkleRoll = 15     # left_ankle_roll_joint
    RightAnkleRoll = 16    # right_ankle_roll_joint
    LeftShoulderYaw = 17   # left_shoulder_yaw_joint
    RightShoulderYaw = 18  # right_shoulder_yaw_joint
    LeftElbow = 19         # left_elbow_joint
    RightElbow = 20        # right_elbow_joint
    LeftWristRoll = 21     # left_wrist_roll_joint
    RightWristRoll = 22    # right_wrist_roll_joint


# Mapping from PyTorch model joint order to MuJoCo joint order
pytorch2mujoco_idx = [
    # PyTorch idx -> MuJoCo idx
    G1MjxJointIndex.LeftHipPitch,      # 0: left_hip_pitch_joint -> LeftHipPitch (0)
    G1MjxJointIndex.RightHipPitch,     # 1: right_hip_pitch_joint -> RightHipPitch (6)
    G1MjxJointIndex.WaistYaw,          # 2: waist_yaw_joint -> WaistYaw (12)
    G1MjxJointIndex.LeftHipRoll,       # 3: left_hip_roll_joint -> LeftHipRoll (1)
    G1MjxJointIndex.RightHipRoll,      # 4: right_hip_roll_joint -> RightHipRoll (7)
    G1MjxJointIndex.LeftHipYaw,        # 5: left_hip_yaw_joint -> LeftHipYaw (2)
    G1MjxJointIndex.RightHipYaw,       # 6: right_hip_yaw_joint -> RightHipYaw (8)
    G1MjxJointIndex.LeftKnee,          # 7: left_knee_joint -> LeftKnee (3)
    G1MjxJointIndex.RightKnee,         # 8: right_knee_joint -> RightKnee (9)
    G1MjxJointIndex.LeftShoulderPitch, # 9: left_shoulder_pitch_joint -> LeftShoulderPitch (13)
    G1MjxJointIndex.RightShoulderPitch,# 10: right_shoulder_pitch_joint -> RightShoulderPitch (18)
    G1MjxJointIndex.LeftAnklePitch,    # 11: left_ankle_pitch_joint -> LeftAnklePitch (4)
    G1MjxJointIndex.RightAnklePitch,   # 12: right_ankle_pitch_joint -> RightAnklePitch (10)
    G1MjxJointIndex.LeftShoulderRoll,  # 13: left_shoulder_roll_joint -> LeftShoulderRoll (14)
    G1MjxJointIndex.RightShoulderRoll, # 14: right_shoulder_roll_joint -> RightShoulderRoll (19)
    G1MjxJointIndex.LeftAnkleRoll,     # 15: left_ankle_roll_joint -> LeftAnkleRoll (5)
    G1MjxJointIndex.RightAnkleRoll,    # 16: right_ankle_roll_joint -> RightAnkleRoll (11)
    G1MjxJointIndex.LeftShoulderYaw,   # 17: left_shoulder_yaw_joint -> LeftShoulderYaw (15)
    G1MjxJointIndex.RightShoulderYaw,  # 18: right_shoulder_yaw_joint -> RightShoulderYaw (20)
    G1MjxJointIndex.LeftElbow,         # 19: left_elbow_joint -> LeftElbow (16)
    G1MjxJointIndex.RightElbow,        # 20: right_elbow_joint -> RightElbow (21)
    G1MjxJointIndex.LeftWristRoll,     # 21: left_wrist_roll_joint -> LeftWristRoll (17)
    G1MjxJointIndex.RightWristRoll,    # 22: right_wrist_roll_joint -> RightWristRoll (22)
]

# Inverse mapping from MuJoCo joint order to PyTorch model order
# This will be automatically generated in init_joint_mappings()
mujoco2pytorch_idx = [0] * 23


def init_joint_mappings():
    """Initialize the inverse mapping from MuJoCo to PyTorch indices."""
    global mujoco2pytorch_idx
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        mujoco2pytorch_idx[mujoco_idx] = pytorch_idx


def remap_pytorch_to_mujoco(pytorch_actions: np.ndarray) -> np.ndarray:
    """Remap actions from PyTorch model joint order to MuJoCo joint order."""
    mujoco_actions = np.zeros_like(pytorch_actions)
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        mujoco_actions[mujoco_idx] = pytorch_actions[pytorch_idx]
    return mujoco_actions


def remap_mujoco_to_pytorch(mujoco_data: np.ndarray) -> np.ndarray:
    """Remap data from MuJoCo joint order to PyTorch model joint order."""
    pytorch_data = np.zeros_like(mujoco_data)
    for pytorch_idx, mujoco_idx in enumerate(pytorch2mujoco_idx):
        pytorch_data[pytorch_idx] = mujoco_data[mujoco_idx]
    return pytorch_data


class TorchController:
  """PyTorch controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
  ):
    self._policy = torch.load(policy_path, weights_only=True)
    self._policy.eval()  # Set to evaluation mode

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)  # In MuJoCo order

    self._counter = 0
    self._n_substeps = n_substeps

    # self._controller = KeyboardController(
    #     vel_scale_x=vel_scale_x,
    #     vel_scale_y=vel_scale_y,
    #     vel_scale_rot=vel_scale_rot,
    # )

    # Initialize joint mappings
    init_joint_mappings()


  def get_obs(self, model, data) -> np.ndarray:
    # Simplified observation: 75 dimensions total
    # projected_gravity (3) + velocity_commands (3) + joint_pos (23) + joint_vel (23) + actions (23)
    
    # Get projected gravity (3 dimensions)
    # imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    # projected_gravity = imu_xmat.T @ np.array([0, 0, -1])
    world_gravity = model.opt.gravity
    world_gravity = world_gravity / np.linalg.norm(world_gravity)  # Normalize
    imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    projected_gravity = imu_xmat.T @ world_gravity
    # Get velocity commands (3 dimensions)
    # velocity_commands = self._controller.get_command()
    velocity_commands = np.array([0.25, 0.0, 0.0])  # Forward velocity command
    
    # Get joint positions and velocities in MuJoCo order, then convert to PyTorch order
    joint_pos_mujoco = data.qpos[7:] - self._default_angles
    joint_vel_mujoco = data.qvel[6:]
    
    # Convert to PyTorch model joint order for the observation
    joint_pos_pytorch = remap_mujoco_to_pytorch(joint_pos_mujoco)
    joint_vel_pytorch = remap_mujoco_to_pytorch(joint_vel_mujoco)
    
    # Last action should also be in PyTorch order for the observation
    # Convert the MuJoCo-ordered last action to PyTorch order
    actions_pytorch = remap_mujoco_to_pytorch(self._last_action)
    
    # Concatenate all observations: 3 + 3 + 23 + 23 + 23 = 75
    obs = np.hstack([
        projected_gravity,
        velocity_commands, 
        joint_pos_pytorch,
        joint_vel_pytorch,
        actions_pytorch,
    ])
    
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      
      # Convert to torch tensor and run inference
      obs_tensor = torch.from_numpy(obs).float()
      
      with torch.no_grad():
        action_tensor = self._policy(obs_tensor)
        pytorch_pred = action_tensor.numpy()  # Actions in PyTorch model joint order

      # Zero out arm control similar to training logic (in PyTorch order)
      # In PyTorch order, arm joints are:
      # 9-10: shoulders, 13-14: shoulder rolls, 17-18: shoulder yaws, 19-20: elbows, 21-22: wrists
      # ZERO_ARM_CONTROL = True  # Set this flag as needed
      # if ZERO_ARM_CONTROL:
      #   # Arm joint indices in PyTorch order
      #   arm_indices = [9, 10, 13, 14, 17, 18, 19, 20, 21, 22]  # All arm joints
      #   pytorch_pred[arm_indices] = 0.0

      # Convert actions from PyTorch order to MuJoCo order
      mujoco_pred = remap_pytorch_to_mujoco(pytorch_pred)

      self._last_action = mujoco_pred.copy()  # Store in MuJoCo order
      # data.ctrl[:] =  self._default_angles

      data.ctrl[:] = mujoco_pred * self._action_scale + self._default_angles


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)


  model = mujoco.MjModel.from_xml_path('./g1_description/scene_mjx_alt.xml')
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  # ctrl_dt = 0.02
  sim_dt = 0.005
  n_substeps = 4
  model.opt.timestep = sim_dt

  # Define default angles based on env.yaml configuration
  # These values are in MuJoCo joint order
  default_angles_config = np.array([
    -0.2,   # LeftHipPitch
    0.0,    # LeftHipRoll
    0.0,    # LeftHipYaw
    0.42,   # LeftKnee
    -0.23,  # LeftAnklePitch
    0.0,    # LeftAnkleRoll
    -0.2,   # RightHipPitch
    0.0,    # RightHipRoll
    0.0,    # RightHipYaw
    0.42,   # RightKnee
    -0.23,  # RightAnklePitch
    0.0,    # RightAnkleRoll
    0.0,    # WaistYaw
    0.35,   # LeftShoulderPitch
    0.16,   # LeftShoulderRoll
    0.0,    # LeftShoulderYaw
    0.87,   # LeftElbow
    0.0,    # LeftWristRoll
    0.35,   # RightShoulderPitch
    -0.16,  # RightShoulderRoll
    0.0,    # RightShoulderYaw
    0.87,   # RightElbow
    0.0,    # RightWristRoll
  ])

  policy = TorchController(
      policy_path=POLICY_PATH,
      default_angles=default_angles_config,
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.0,
      vel_scale_y=1.0,
      vel_scale_rot=1.0,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
