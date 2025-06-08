import subprocess
import os
import itertools
from sweep_analyzer import analyze_sweep_results

# Parameter ranges for the sweep
AIR_TIME_WEIGHTS = [3.0, 5.0]
USE_DATA_AUGMENTATION = [False, True]
BOTH_FEET_AIR_WEIGHTS = [0.0, 1e-2, 1e-1, 1, 5]
MIRROR_LOSS_COEFFS = [0.5, 1.0]

# Experiment configuration
EXPERIMENT_NAME = "g1_23dof_sweep_v7"  # Must match cfg 

def run_command(command_args, description="Running command"):
    """Helper function to execute shell commands and stream output to CLI."""
    print(f"\n--- {description} ---")
    print(f"Command: {' '.join(command_args)}")
    try:
        # Popen allows streaming output
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream output line by line
        for line in process.stdout:
            print(line, end='') # `end=''` prevents extra newlines

        process.wait() # Wait for the process to complete

        if process.returncode != 0:
            print(f"\nError: Command exited with non-zero status {process.returncode}")
            exit(1) # Exit if a command fails
        else:
            print("\nCommand completed successfully.")

    except FileNotFoundError:
        print(f"Error: Command not found. Make sure '{command_args[0]}' is in your PATH or correctly specified.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

def main():
    # Define base commands - updated to use EXPERIMENT_NAME
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--task=Loco", "--headless"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task=Loco", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Generate parameter combinations
    parameter_combinations = []
    
    for air_time_weight in AIR_TIME_WEIGHTS:
        for use_data_aug in USE_DATA_AUGMENTATION:
            for both_feet_air_weight in BOTH_FEET_AIR_WEIGHTS:
                for mirror_loss_coeff in MIRROR_LOSS_COEFFS:
                    parameter_combinations.append((air_time_weight, use_data_aug, both_feet_air_weight, mirror_loss_coeff))
    
    print(f"Starting parameter sweep: {EXPERIMENT_NAME}")
    print(f"Generated {len(parameter_combinations)} parameter combinations to test.")
    print(f"Air time weights: {AIR_TIME_WEIGHTS}")
    print(f"Use data augmentation: {USE_DATA_AUGMENTATION}")
    print(f"Both feet air weights: {BOTH_FEET_AIR_WEIGHTS}")
    print(f"Mirror loss coeffs: {MIRROR_LOSS_COEFFS}")
    
    for i, (air_time_weight, use_data_aug, both_feet_air_weight, mirror_loss_coeff) in enumerate(parameter_combinations):
        description = f"Set {i+1}/{len(parameter_combinations)}: air_time={air_time_weight}, data_aug={use_data_aug}, both_feet_air={both_feet_air_weight}, mirror_coeff={mirror_loss_coeff}"
        
        print(f"\n{'='*80}\nStarting {description}\n{'='*80}")

        # Construct parameter arguments
        train_args = [
            f"env.rewards.feet_air_time.weight={air_time_weight}",
            f"agent.algorithm.symmetry_cfg.use_data_augmentation={use_data_aug}",
            f"agent.algorithm.symmetry_cfg.use_mirror_loss=True",
            f"agent.algorithm.symmetry_cfg.mirror_loss_coeff={mirror_loss_coeff}",
            f"env.rewards.both_feet_air.weight={both_feet_air_weight}"
        ]

        # Construct the full train command
        full_train_cmd = TRAIN_BASE_CMD + train_args
        run_command(full_train_cmd, f"Training for {description}")

        # Construct the full play command (no extra args needed for play usually)
        full_play_cmd = PLAY_BASE_CMD
        run_command(full_play_cmd, f"Playing for {description}")

    print(f"\n🎉 All {len(parameter_combinations)} parameter combinations finished!")
    
    # Automatically analyze results
    print(f"\n🔍 Starting automatic analysis of sweep results...")
    
    try:
        results = analyze_sweep_results(
            experiment_name=EXPERIMENT_NAME,
            air_time_weights=AIR_TIME_WEIGHTS,
            use_data_augmentation=USE_DATA_AUGMENTATION,
            both_feet_air_weights=BOTH_FEET_AIR_WEIGHTS,
            mirror_loss_coeffs=MIRROR_LOSS_COEFFS
        )
        
        if results and results['success']:
            print(f"\n🎊 SWEEP COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved:")
            print(f"   📄 Parameter guide: {results['param_guide_file']}")
            print(f"   🎬 Basic concatenated video: {results['video_file']}")
            if results.get('labeled_video_file'):
                print(f"   🏷️  Labeled concatenated video: {results['labeled_video_file']}")
            print(f"   🗺️  Video mapping: {results['video_mapping_file']}")
        else:
            print(f"\n⚠️  Sweep completed but analysis had issues. Check the output above.")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print(f"   You can manually run analysis later with:")
        print(f"   python sweep_analyzer.py")

if __name__ == "__main__":
    main() 
    import os
    import time
    
    print("\n⏳ Waiting 5 seconds before suspend...")
    time.sleep(5)
    print("💤 Suspending computer...")
    os.system("systemctl suspend")