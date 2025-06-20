import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"assets/g1_23dof.usd",
        usd_path=f"assets/g1_23dof_simple.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/G1_with_hand/g1_29dof_with_hand_rev_1_0.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
            # fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
            },
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 60.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_pitch_joint": 60.0,
                ".*_knee_joint": 100.0,
                "waist_yaw_joint": 60.0,
            },
            damping={
                ".*_hip_yaw_joint": 1.0,
                ".*_hip_roll_joint": 1.0,
                ".*_hip_pitch_joint": 1.0,
                ".*_knee_joint": 2.0,
                "waist_yaw_joint": 1.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.01017752004,
                ".*_hip_roll_joint": 0.025101925,
                ".*_hip_pitch_joint": 0.01017752004,
                ".*_knee_joint": 0.025101925,
                "waist_yaw_joint": 0.01017752004,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=50.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=1.0,
            armature=0.00721945,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit=25.0,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.003609725,
        ),
    },
)