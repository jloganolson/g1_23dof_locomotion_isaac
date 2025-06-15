
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# from g1_23dof_locomotion_isaac.tasks.manager_based.g1_23dof_locomotion_isaac.g1_23dof_locomotion_isaac_env_cfg import G123dofLocomotionIsaacEnvCfg
# from isaaclab.envs import ManagerBasedRLEnv
from g1_23dof_locomotion_isaac.tasks.manager_based.g1_23dof_locomotion_isaac.g1 import G1_CFG

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
from isaaclab.sensors import  ImuCfg, OffsetCfg

import torch
import numpy as np

class TestSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )




    # robot
    RobotA = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotA")
    RobotB = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotB")
    



def mirror_joint_tensor(original: torch.Tensor, mirrored: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Mirror a tensor of joint values by swapping left/right pairs and inverting yaw/roll joints.
    
    Args:
        original: Input tensor of shape [..., num_joints] where num_joints is 23
        mirrored: Output tensor of same shape to store mirrored values
        offset: Optional offset to add to indices if tensor has additional dimensions
        
    Returns:
        Mirrored tensor with same shape as input
    """
    # Define pairs of indices to swap (left/right pairs)
    swap_pairs = [
        (0 + offset, 1 + offset),   # hip_pitch
        (3 + offset, 4 + offset),   # hip_roll
        (5 + offset, 6 + offset),   # hip_yaw
        (7 + offset, 8 + offset),   # knee
        (9 + offset, 10 + offset),  # shoulder_pitch
        (11 + offset, 12 + offset), # ankle_pitch
        (13 + offset, 14 + offset), # shoulder_roll
        (15 + offset, 16 + offset), # ankle_roll
        (17 + offset, 18 + offset), # shoulder_yaw
        (19 + offset, 20 + offset), # elbow
        (21 + offset, 22 + offset)  # wrist_roll
    ]
    
    # Define indices that need to be inverted (yaw/roll joints)
    invert_indices = [
        2 + offset,   # waist_yaw
        3 + offset,   # left_hip_roll
        4 + offset,   # right_hip_roll
        5 + offset,   # left_hip_yaw
        6 + offset,   # right_hip_yaw
        13 + offset,  # left_shoulder_roll
        14 + offset,  # right_shoulder_roll
        15 + offset,  # left_ankle_roll
        16 + offset,  # right_ankle_roll
        17 + offset,  # left_shoulder_yaw
        18 + offset,  # right_shoulder_yaw
        21 + offset,  # left_wrist_roll
        22 + offset   # right_wrist_roll
    ]
    
    # First copy non-swapped, non-inverted values
    non_swap_indices = [i for i in range(original.shape[-1]) if i not in [idx for pair in swap_pairs for idx in pair]]
    mirrored[..., non_swap_indices] = original[..., non_swap_indices]
    
    # Swap left/right pairs
    for left_idx, right_idx in swap_pairs:
        mirrored[..., left_idx] = original[..., right_idx]
        mirrored[..., right_idx] = original[..., left_idx]
    
    # Invert yaw/roll joints
    mirrored[..., invert_indices] = -mirrored[..., invert_indices]
    

def mirror_actions(actions):
    if actions is None:
        return None

    _actions = torch.clone(actions)
    flip_actions = torch.zeros_like(_actions)
    mirror_joint_tensor(_actions, flip_actions)
    return torch.vstack((_actions, flip_actions))

def scene_reset(scene: InteractiveScene):
    root_robotA_state = scene["RobotA"].data.default_root_state.clone()
    root_robotA_state[:, :3] += scene.env_origins
    root_robotB_state = scene["RobotB"].data.default_root_state.clone()
    root_robotB_state[:, :3] += scene.env_origins + torch.tensor([0.0, 1.0, 0.0], device="cuda")

    # copy the default root state to the sim for RobotA and RobotB's orientation and velocity
    scene["RobotA"].write_root_pose_to_sim(root_robotA_state[:, :7])
    scene["RobotA"].write_root_velocity_to_sim(root_robotA_state[:, 7:])
    scene["RobotB"].write_root_pose_to_sim(root_robotB_state[:, :7])
    scene["RobotB"].write_root_velocity_to_sim(root_robotB_state[:, 7:])

    # copy the default joint states to the sim
    joint_pos, joint_vel = (
        scene["RobotA"].data.default_joint_pos.clone(),
        scene["RobotA"].data.default_joint_vel.clone(),
    )
    scene["RobotA"].write_joint_state_to_sim(joint_pos, joint_vel)
    joint_pos, joint_vel = (
        scene["RobotB"].data.default_joint_pos.clone(),
        scene["RobotB"].data.default_joint_vel.clone(),
    )





    scene["RobotB"].write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()



# dp.undesired_contacts,
#     #     weight=-1.0,
#     #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    scene_reset(scene)
    count = 0
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()
    #do a random one based on a ROS event
    while simulation_app.is_running():


        count += 1
        if count % 100 == 0:
            print("step: ", count)
            default_joint_pos = scene["RobotA"].data.default_joint_pos.squeeze(0)
            # action_b = scene["RobotB"].data.default_joint_pos

            
            lower = scene["RobotA"].data.joint_pos_limits[0, :, 0]
            upper = scene["RobotA"].data.joint_pos_limits[0, :, 1]

   
            mean = (upper + lower) / 2
            std = (upper - lower) / 6  # Using 6 sigma to keep most values within bounds
            random_pos = torch.normal(mean, std)
            random_pos = torch.clamp(random_pos, lower, upper)
            mirrored_actions = mirror_actions(random_pos)

            # Create a mask for specific indices to use mirrored actions
            mask_indices = [3,4]  # Example indices to use mirrored actions
            mask = torch.zeros_like(random_pos)
            mask[mask_indices] = 1.0

            gravity = scene["RobotA"].data.projected_gravity_b
 
            # # Combine default and mirrored actions based on mask for each robot
            # combined_actions_a = torch.where(mask == 1.0,
            #                               mirrored_actions[0],
            #                               default_joint_pos)
            
       

            # combined_actions_b = torch.where(mask == 1.0,
            #                               mirrored_actions[1],
            #                               default_joint_pos)

            # print("\nActions for mask indices:")
            # for idx in mask_indices:
            #     print(f"Index {idx}:")
            #     print(f"  Action A: {combined_actions_a[idx]}")
            #     print(f"  Action B: {combined_actions_b[idx]}")
            
            # Apply the combined actions to each robot
            # scene["RobotA"].set_joint_position_target(combined_actions_a)
            # scene["RobotB"].set_joint_position_target(combined_actions_b)
           
            # marker_orientations = gravity_quat
            scene["RobotA"].set_joint_position_target(mirrored_actions[0])
            scene["RobotB"].set_joint_position_target(mirrored_actions[1])
            # my_visualizer.visualize(marker_locations, marker_orientations, marker_indices=[0])
            draw_interface.clear_lines()
            line_start_points = [
                [0, 0, 1.],
            ]
            line_end_points = [
                [0, 1., 1],
            ]
            line_colors = [(1, 0, 0, 1)] * 1
            line_sizes = [2.0] * 1
            draw_interface.draw_lines(line_start_points, line_end_points, line_colors, line_sizes)

            scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()