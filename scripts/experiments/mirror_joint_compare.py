
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
import torch

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
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
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
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state...")

        # # drive around
        # if count % 100 < 75:
        #     # Drive straight by setting equal wheel velocities
        #     action = torch.Tensor([[10.0, 10.0]])
        # else:
        #     # Turn by applying different velocities
        #     action = torch.Tensor([[5.0, -5.0]])

        # scene["Jetbot"].set_joint_velocity_target(action)

        # # wave
        # wave_action = scene["Dofbot"].data.default_joint_pos
        # wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        # scene["Dofbot"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
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