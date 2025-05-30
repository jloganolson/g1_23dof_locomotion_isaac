import subprocess
import os

def run_command(command_args, description="Running command"):
    """Helper function to execute shell commands."""
    print(f"\n--- {description} ---")
    print(f"Command: {' '.join(command_args)}")
    try:
        # Use subprocess.run for robust command execution
        # check=True will raise a CalledProcessError if the command returns a non-zero exit code
        # text=True decodes stdout/stderr as text
        subprocess.run(command_args, check=True, text=True, capture_output=True)
        print("Command completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        exit(1) # Exit if a command fails
    except FileNotFoundError:
        print(f"Error: Command not found. Make sure '{command_args[0]}' is in your PATH or correctly specified.")
        exit(1)

def main():
    # --- Conda Environment Activation (Crucial for running from a non-initialized shell) ---
    # This block ensures that the Python script runs within the correct Conda environment
    # even if the shell it's launched from hasn't sourced Conda.
    # It checks if the environment is already active, and activates it if not.
    # This might look a bit different from the shell script, as we're managing the environment
    # from inside Python for subprocess calls.

    # Option 1: Assume `python` is already in the right env or rely on `conda run` (simpler)
    # This is often the easiest way if you want the script itself to manage environment activation.
    # We'll use this approach for simplicity in the examples below, assuming
    # 'conda activate env_isaaclab' has been run *before* executing this script,
    # OR we use `conda run -n env_isaaclab python run_experiments.py`.

    # Option 2: Programmatically activate Conda (more complex for general use)
    # If this script *must* activate Conda itself regardless of how it's launched,
    # you'd need to find the conda.sh and source it, then restart the script in the env.
    # For typical experiment management, it's simpler to ensure the script is run from
    # the correct environment or use `conda run`.
    # Let's assume you'll run this script using `conda run` or after `conda activate`.

    # Define base commands
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--task=Loco", "--headless"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task=Loco", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Define your parameter sets as a list of dictionaries
    # Each dictionary represents a set of command-line arguments
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