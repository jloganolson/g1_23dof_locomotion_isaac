import subprocess
import os

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

# ... (rest of your main function remains the same) ...
# Define base commands, parameter_sets, and loop through them as before.
# The 'main' function and `if __name__ == "__main__":` block are unchanged.

def main():
    # Define base commands
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--task=Loco", "--headless"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task=Loco", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Define your parameter sets as a list of dictionaries
    parameter_sets = [
        # Parameter Set 1
        {"description": "Parameter Set 1",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=True",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
                        "env.rewards.feet_slide.weight=-0.1"],
         "play_args": []}, # No extra play args usually

        # Parameter Set 2
        {"description": "Parameter Set 2",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=False",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
                        "env.rewards.feet_slide.weight=-0.1"],
         "play_args": []},

        # Parameter Set 3
        {"description": "Parameter Set 3",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=False",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
                        "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.5"],
         "play_args": []},

        # Parameter Set 4
        {"description": "Parameter Set 4",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=True",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
                        "env.rewards.feet_slide.weight=-0.5"],
         "play_args": []},

        # Parameter Set 5
        {"description": "Parameter Set 5",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=False",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
                        "env.rewards.feet_slide.weight=-0.5"],
         "play_args": []},

        # Parameter Set 6
        {"description": "Parameter Set 6",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=False",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=False",
                        "env.rewards.feet_slide.weight=-0.5"],
         "play_args": []},

        # Parameter Set 7
        {"description": "Parameter Set 7",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=False",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=True",
                        "agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.1"],
         "play_args": []},

        # Parameter Set 8
        {"description": "Parameter Set 8",
         "train_args": ["agent.algorithm.symmetry_cfg.use_data_augmentation=False",
                        "agent.algorithm.symmetry_cfg.use_mirror_loss=False"],
         "play_args": []},
    ]

    for i, params in enumerate(parameter_sets):
        print(f"\n{'='*50}\nStarting {params['description']} (Set {i+1}/{len(parameter_sets)})\n{'='*50}")

        # Construct the full train command
        full_train_cmd = TRAIN_BASE_CMD + params["train_args"]
        run_command(full_train_cmd, f"Training for {params['description']}")

        # Construct the full play command
        full_play_cmd = PLAY_BASE_CMD + params["play_args"]
        run_command(full_play_cmd, f"Playing for {params['description']}")

    print("\nAll parameter sets finished.")

if __name__ == "__main__":
    main()