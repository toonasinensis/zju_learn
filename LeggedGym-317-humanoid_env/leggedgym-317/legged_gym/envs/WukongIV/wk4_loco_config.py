from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, BaseConfig
from legged_gym.utils.reward import GaussianRewardParam, CauchyRewardParam
import time

class HumanLocoCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_history_frames = 50
        history_stride = 1
        num_observations = 48
        num_observations_history = num_observations * num_history_frames
        num_actions = 12
        episode_length_s = 20  # episode length in seconds
        num_privileged_obs = num_observations + 1 + 18 + 81  # base vel, base height and foot height
        estimation_terms = ["vel"]  # est v+z
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm

    class terrain:
        measure_heights = True
        mesh_type = 'plane'  # none, plane, heightfield or trimesh
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        measured_foot_points_x = [-0.1, 0., 0.1]  # 1mx1.6m rectangle (without center line)
        measured_foot_points_y = [-0.1, 0., 0.1]  # 1mx1.6m rectangle (without center line)
        curriculum = False
        robot_move = True  # Whether the robot moves (to change the curriculum)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.1, 0.1, 0.3]
        static_friction = 0.75
        dynamic_friction = 0.7
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        terrain_length = 8.
        terrain_width = 8.
        border_length = 10
        border_width = 10
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        restitution = 0.
        # Heightfeild only:
        heightfeild_range = [-0., 0.]
        rough_slope_range = [-0.1, 0.1]
        heightfeild_resolution = 0.005
        measure_foot_heights = False
        selected = True  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        # default: Vx, Vy, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # for humanoid, expand for arm operation
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error        curriculum = False
        max_curriculum = 1.

        class ranges:
            lin_vel_x = [1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4]  # min max [rad/s]
            heading = [-1.57, 1.57]

    class init_state:
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        pos = [0.0, 0.0, 0.72]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Waist': 0.,
            'Shoulder_Z_R': -0.1,
            'Shoulder_X_R': 0.,
            'Shoulder_Y_R': 0.,
            'Elbow_R': 0.525,
            'Shoulder_Z_L': -0.1,
            'Shoulder_X_L': 0.,
            'Shoulder_Y_L': 0.,
            'Elbow_L': 0.525,

            'Hip_Z_R': 0,
            'Hip_X_R': -0.08,
            'Hip_Y_R': -0.4,
            'Knee_R': 0.95,
            'Ankle_Y_R': -0.44,
            'Ankle_X_R': 0.04,
            'Hip_Z_L': 0,
            'Hip_X_L': -0.08,
            'Hip_Y_L': -0.4,
            'Knee_L': 0.95,
            'Ankle_Y_L': -0.44,
            'Ankle_X_L': 0.04
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'Waist': 200.0, 'Shoulder_Z': 30, 'Shoulder_X': 40, 'Shoulder_Y': 60, 'Elbow': 30,
                     'Hip_Z': 100, 'Hip_X': 80, 'Hip_Y': 240, 'Knee': 240, 'Ankle_Y': 100, 'Ankle_X': 100
                     }  # [N*m/rad]
        # orgin PD of ankle: P: 'Ankle_Y': 50, 'Ankle_X': 50; D:'Ankle_Y': 0.5, 'Ankle_X': 0.3
        damping = {'Waist': 4., 'Shoulder_Z': 0.5, 'Shoulder_X': 0.5, 'Shoulder_Y': 0.8, 'Elbow': 0.5,
                   'Hip_Z': 1, 'Hip_X': 1, 'Hip_Y': 2.5, 'Knee': 4, 'Ankle_Y': 2, 'Ankle_X': 2
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 0.5
        action_scale = {'Waist': 0.1, 'Shoulder_Z': 0.05, 'Shoulder_X': 0.02, 'Shoulder_Y': 0.5, 'Elbow': 0.3,
                        'Hip_Z': 0.5, 'Hip_X': 0.5, 'Hip_Y': 0.5, 'Knee': 0.5, 'Ankle_Y': 0.5, 'Ankle_X': 0.5}

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/WukongIV/urdf/WuKongIV_preciseMass.urdf'
        name = "Wukong4"
        foot_name = 'FOOT'
        terminate_after_contacts_on = ['TORSO', 'UPPERARM', 'ILIUM', 'ISCHIUM', 'THIGH']
        penalize_contacts_on = []

        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2., 12.5]
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = True
        Kd_factor_range = [0.9, 1.1]
        rand_interval_s = 8
        push_robots = True
        push_interval_s = 2.5
        # Unit of interval is sec
        max_push_vel_xy = 0.5
        randomize_init_state = True
        randomize_lag_timesteps = True
        lag_timesteps = 2
        randomize_init_dof = True

    class rewards:
        class item:
            lin_Vel = GaussianRewardParam(20, 0.4, 0)
            ang_Vel = GaussianRewardParam(5, 0.4, 0)
            bRot = GaussianRewardParam(5, 0.1, 0)
            bHgt = GaussianRewardParam(2, 0.0707, 0)
            # cycle_time = 0.64                # sec

            # cotr = CauchyRewardParam(0.05, 0.2, 1, 0)
            # jVel = CauchyRewardParam(0.05, 16, 2, 0)
        cycle_time = 0.64                # sec

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        #TODO:
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.71
        foot_height_target = 0.12
        max_contact_force = 600.  # forces above this value are penalized
        mean_vel_window = 1.0  # window length for computing average body velocity (in sec)

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.166667
            dof_pos = 1.0
            dof_vel = 0.1
            height_measurements = 1.
            base_height = 30.
            foot_height = 30.

        clip_observations = 100.
        clip_actions = 10.

        whole_body_dof_order = [
            'Waist',
            'Shoulder_Z_R', 'Shoulder_X_R', 'Shoulder_Y_R', 'Elbow_R',
            'Shoulder_Z_L', 'Shoulder_X_L', 'Shoulder_Y_L', 'Elbow_L',
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

        obs_dof_order = [
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

        action_dof_order = [
            'Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_R', 'Ankle_Y_R', 'Ankle_X_R',
            'Hip_Z_L', 'Hip_X_L', 'Hip_Y_L', 'Knee_L', 'Ankle_Y_L', 'Ankle_X_L'
        ]

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurements = 0.05

    class viewer:
        ref_env = 0
        pos = [-2, -2, 6]  # [m]
        lookat = [0., 0, 3.]  # [m]

    class sim:
        dt = 0.001
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HumanLocoCfgPPO:
    seed = time.time()
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        encoder_hidden_dims = [1024, 256, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        latent_dims = 16 + 3 # 16: implicit encoding; 3: velocity estimation;
        estimation_dims = {"vel": 3, "implicit": 16}

    class runner:
        policy_class_name = 'PrivilegedActorCritic'
        algorithm_class_name = 'PPO'

        max_iterations = 700  # number of policy updates
        num_steps_per_env = 48  # per iteration
        run_name = 'test'
        experiment_name = 'rough_wk4'
        save_interval = 100  # check for potential saves every this many iterations
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        save_items = [f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wk4_leg_config.py",
                      f"{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/WukongIV/wukong4_leg.py"]

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * n steps / n minibatches
        learning_rate = 5.e-4  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 0.5
        est_loss_coeff = {"vel": 40}
        obs_mse_coeff = 2
        vae_kl_coeff = 50
