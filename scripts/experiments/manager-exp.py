
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from g1_23dof_locomotion_isaac.tasks.manager_based.g1_23dof_locomotion_isaac.g1_23dof_locomotion_isaac_env_cfg import G123dofLocomotionIsaacEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

def main():
    print("Hello, World!")
    env_cfg = G123dofLocomotionIsaacEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    print(robot.joint_names)

if __name__ == "__main__":
    main()