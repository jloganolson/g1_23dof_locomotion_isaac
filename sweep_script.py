import subprocess
import os
import itertools

# Parameter ranges for the sweep
AIR_TIME_WEIGHTS = [ 1.0, 5.0, 20.0]
TORQUE_WEIGHTS = [ -1.0e-5, -1.0e-4, -1.0e-3]
ACC_WEIGHTS = [-1.0e-7, -5.0e-6, -5.0e-5]
# AIR_TIME_WEIGHTS = [.25, 1.0, 5.0, 20.0]
# TORQUE_WEIGHTS = [-2.0e-6, -1.0e-5, -1.0e-4, -1.0e-3]
# ACC_WEIGHTS = [-2.5e-7, -1.0e-7, -5.0e-6, -5.0e-5]

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
    # Define base commands
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--task=Loco", "--headless"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task=Loco", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(AIR_TIME_WEIGHTS, TORQUE_WEIGHTS, ACC_WEIGHTS))
    
    print(f"Generated {len(parameter_combinations)} parameter combinations to test.")
    print(f"Air time weights: {AIR_TIME_WEIGHTS}")
    print(f"Torque weights: {TORQUE_WEIGHTS}")
    print(f"Acceleration weights: {ACC_WEIGHTS}")
    
    for i, (air_time_weight, torque_weight, acc_weight) in enumerate(parameter_combinations):
        description = f"Set {i+1}/{len(parameter_combinations)}: air_time={air_time_weight}, torque={torque_weight:.1e}, acc={acc_weight:.1e}"
        
        print(f"\n{'='*80}\nStarting {description}\n{'='*80}")

        # Construct parameter arguments
        train_args = [
            f"env.rewards.feet_air_time.weight={air_time_weight}",
            f"env.rewards.dof_torques_l2.weight={torque_weight}",
            f"env.rewards.dof_acc_l2.weight={acc_weight}"
        ]

        # Construct the full train command
        full_train_cmd = TRAIN_BASE_CMD + train_args
        run_command(full_train_cmd, f"Training for {description}")

        # Construct the full play command (no extra args needed for play usually)
        full_play_cmd = PLAY_BASE_CMD
        run_command(full_play_cmd, f"Playing for {description}")

    print(f"\nAll {len(parameter_combinations)} parameter combinations finished.")

if __name__ == "__main__":
    main()