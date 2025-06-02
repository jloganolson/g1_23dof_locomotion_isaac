import subprocess
import os
import itertools
from sweep_analyzer import analyze_sweep_results

# Parameter ranges for the sweep
AIR_TIME_WEIGHTS = [1.0, 3.0, 5.0]
USE_DATA_AUGMENTATION = [False, True]
USE_MIRROR_LOSS = [False, True]
MIRROR_LOSS_COEFFS = [0.5, 1.0]  # Only used when USE_MIRROR_LOSS is True

# Experiment configuration
EXPERIMENT_NAME = "g1_23dof_sweep_v5"  # Update version for new sweep

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
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--task=Loco", "--headless", f"--experiment={EXPERIMENT_NAME}"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task=Loco", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Generate parameter combinations
    parameter_combinations = []
    
    for air_time_weight in AIR_TIME_WEIGHTS:
        for use_data_aug in USE_DATA_AUGMENTATION:
            for use_mirror_loss in USE_MIRROR_LOSS:
                if use_mirror_loss:
                    # If mirror loss is enabled, try different coefficients
                    for mirror_loss_coeff in MIRROR_LOSS_COEFFS:
                        parameter_combinations.append((air_time_weight, use_data_aug, use_mirror_loss, mirror_loss_coeff))
                else:
                    # If mirror loss is disabled, coefficient doesn't matter (use None as placeholder)
                    parameter_combinations.append((air_time_weight, use_data_aug, use_mirror_loss, None))
    
    print(f"Starting parameter sweep: {EXPERIMENT_NAME}")
    print(f"Generated {len(parameter_combinations)} parameter combinations to test.")
    print(f"Air time weights: {AIR_TIME_WEIGHTS}")
    print(f"Use data augmentation: {USE_DATA_AUGMENTATION}")
    print(f"Use mirror loss: {USE_MIRROR_LOSS}")
    print(f"Mirror loss coeffs (when enabled): {MIRROR_LOSS_COEFFS}")
    
    for i, (air_time_weight, use_data_aug, use_mirror_loss, mirror_loss_coeff) in enumerate(parameter_combinations):
        if mirror_loss_coeff is not None:
            description = f"Set {i+1}/{len(parameter_combinations)}: air_time={air_time_weight}, data_aug={use_data_aug}, mirror_loss={use_mirror_loss}, mirror_coeff={mirror_loss_coeff}"
        else:
            description = f"Set {i+1}/{len(parameter_combinations)}: air_time={air_time_weight}, data_aug={use_data_aug}, mirror_loss={use_mirror_loss}"
        
        print(f"\n{'='*80}\nStarting {description}\n{'='*80}")

        # Construct parameter arguments
        train_args = [
            f"env.rewards.feet_air_time.weight={air_time_weight}",
            f"agent.algorithm.symmetry_cfg.use_data_augmentation={use_data_aug}",
            f"agent.algorithm.symmetry_cfg.use_mirror_loss={use_mirror_loss}"
        ]
        
        # Add mirror loss coefficient only if mirror loss is enabled
        if use_mirror_loss and mirror_loss_coeff is not None:
            train_args.append(f"agent.algorithm.symmetry_cfg.mirror_loss_coeff={mirror_loss_coeff}")

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
            use_mirror_loss=USE_MIRROR_LOSS,
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