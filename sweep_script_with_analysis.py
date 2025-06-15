import subprocess
import itertools
from sweep_analyzer import analyze_sweep_results
import os
from datetime import datetime



# Experiment configuration
EXPERIMENT_NAME = "g1_23dof_sweep_v16"  # Must match cfg 

# =============================================================================
# PARAMETER SWEEP CONFIGURATION - CENTRALIZED
# =============================================================================
# Define which parameters to sweep (as lists) and which to keep fixed (as single values)
SWEEP_CONFIG = {
    # Parameters to sweep - each should be a list of values to test
    "SWEEP_PARAMS": {
        # "agent.algorithm.symmetry_cfg.use_mirror_loss": [True, False],
        # "env.rewards.feet_air_time.params.threshold": [0.1, 0.2],
        "env.rewards.feet_air_time.weight": [1.0,3.0,5.0],
        "env.rewards.both_feet_air.weight": [-0.1, -0.5, 0.],
        "env.rewards.action_rate_l2.weight": [-0.005,-0.01],
        "env.rewards.joint_deviation_arms.weight": [-0.1,-.2],

        # "env.rewards.dof_acc_l2.weight": [-5.0e-7, -1.0e-6],
        # "env.rewards.track_lin_vel_xy_exp.weight": [1.,2.],
        # "env.rewards.track_ang_vel_z_exp.weight": [2.,1.],
    },

}
# =============================================================================

def log_parameter_combination(combination, run_number, total_runs, log_file, status="STARTED", error=None):
    """Log parameter combination to a text file with status tracking."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\nRun #{run_number}/{total_runs} - {timestamp} - {status}\n")
        for param_name, param_value in combination.items():
            f.write(f"{param_name}: {param_value}\n")
        if error:
            f.write(f"ERROR: {error}\n")
        f.write("-" * 50 + "\n")

def generate_parameter_combinations():
    """Generate all combinations of sweep parameters."""
    sweep_params = SWEEP_CONFIG["SWEEP_PARAMS"]
    
    if not sweep_params:
        return [{}]  # Return single empty combination if no sweep params
    
    # Get parameter names and their possible values
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)
    
    return combinations

def build_train_args(sweep_params):
    """Build training arguments from sweep and fixed parameters."""
    train_args = []
    
    # Add sweep parameters
    for param_name, param_value in sweep_params.items():
        train_args.append(f"{param_name}={param_value}")
    
   
    
    return train_args

def get_combination_description(sweep_params, combination_num, total_combinations):
    """Generate a description for the current parameter combination."""
    if not sweep_params:
        return f"Set {combination_num}/{total_combinations}: (fixed parameters only)"
    
    param_strs = [f"{param_name.split('.')[-1]}={param_value}" 
                  for param_name, param_value in sweep_params.items()]
    param_description = ", ".join(param_strs)
    
    return f"Set {combination_num}/{total_combinations}: {param_description}"

def run_command(command_args, description="Running command", prefix=None, log_file=None, run_number=None, total_runs=None, combination=None):
    """Helper function to execute shell commands and stream output to CLI."""
    print(f"\n--- {description} ---")
    print(f"Command: {' '.join(command_args)}")
    try:
        # Popen allows streaming output
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream output line by line
        for line in process.stdout:
            if prefix:
                # Add prefix to each line, preserving original formatting
                print(f"[{prefix}] {line}", end='')
            else:
                print(line, end='') # `end=''` prevents extra newlines

        process.wait() # Wait for the process to complete

        if process.returncode != 0:
            error_msg = f"Command exited with non-zero status {process.returncode}"
            print(f"\nError: {error_msg}")
            if log_file and run_number and total_runs and combination:
                log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg)
            exit(1) # Exit if a command fails
        else:
            print("\nCommand completed successfully.")
            if log_file and run_number and total_runs and combination:
                log_parameter_combination(combination, run_number, total_runs, log_file, status="COMPLETED")

    except FileNotFoundError:
        error_msg = f"Command not found. Make sure '{command_args[0]}' is in your PATH or correctly specified."
        print(f"Error: {error_msg}")
        if log_file and run_number and total_runs and combination:
            log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg)
        exit(1)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(f"Error: {error_msg}")
        if log_file and run_number and total_runs and combination:
            log_parameter_combination(combination, run_number, total_runs, log_file, status="FAILED", error=error_msg)
        exit(1)

def main():
    # Define base commands - updated to use EXPERIMENT_NAME
    TRAIN_BASE_CMD = ["python", "scripts/rsl_rl/train.py", "--task=Loco", "--headless"]
    PLAY_BASE_CMD = ["python", "scripts/rsl_rl/play.py", "--task=Loco", "--headless", "--video", "--video_length", "200", "--enable_cameras"]

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"sweep_log_{EXPERIMENT_NAME}_{timestamp}.txt"
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Parameter Sweep Log - {EXPERIMENT_NAME}\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nSweep Configuration:\n")
        for param_name, param_values in SWEEP_CONFIG["SWEEP_PARAMS"].items():
            f.write(f"{param_name}: {param_values}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\nRun Status Summary:\n")
        f.write("==================\n")

    # Generate parameter combinations
    parameter_combinations = generate_parameter_combinations()
    
    print(f"Starting parameter sweep: {EXPERIMENT_NAME}")
    print(f"Generated {len(parameter_combinations)} parameter combinations to test.")
    print(f"Logging to: {log_file}")
    
    # Print sweep configuration
    print(f"\nSweep parameters:")
    for param_name, param_values in SWEEP_CONFIG["SWEEP_PARAMS"].items():
        print(f"  {param_name}: {param_values}")
    
    for i, combination in enumerate(parameter_combinations):
        description = get_combination_description(combination, i+1, len(parameter_combinations))
        run_prefix = f"Run {i+1}/{len(parameter_combinations)}"
        
        # Log the start of the parameter combination
        log_parameter_combination(combination, i+1, len(parameter_combinations), log_file, status="STARTED")
        
        print(f"\n{'='*80}\nStarting {description}\n{'='*80}")

        # Construct parameter arguments
        train_args = build_train_args(combination)

        # Construct the full train command
        full_train_cmd = TRAIN_BASE_CMD + train_args
        run_command(full_train_cmd, f"Training for {description}", prefix=run_prefix, 
                   log_file=log_file, run_number=i+1, total_runs=len(parameter_combinations), 
                   combination=combination)

        # Construct the full play command (no extra args needed for play usually)
        full_play_cmd = PLAY_BASE_CMD
        run_command(full_play_cmd, f"Playing for {description}", prefix=run_prefix,
                   log_file=log_file, run_number=i+1, total_runs=len(parameter_combinations),
                   combination=combination)

    print(f"\n🎉 All {len(parameter_combinations)} parameter combinations finished!")
    
    # Automatically analyze results
    print(f"\n🔍 Starting automatic analysis of sweep results...")
    
    # Log analysis start
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Analysis Results\n")
        f.write("================\n")
        f.write(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Simple analysis with just the experiment name
        results = analyze_sweep_results(experiment_name=EXPERIMENT_NAME)
        
        # Log analysis results
        with open(log_file, 'a') as f:
            if results and results['success']:
                f.write("\n✅ Analysis completed successfully!\n")
                f.write("Generated files:\n")
                f.write(f"  - Experiment guide: {results['experiment_guide_file']}\n")
                f.write(f"  - Basic concatenated video: {results['video_file']}\n")
                if results.get('labeled_video_file'):
                    f.write(f"  - Labeled concatenated video: {results['labeled_video_file']}\n")
                f.write(f"  - Video mapping: {results['video_mapping_file']}\n")
            else:
                f.write("\n⚠️ Analysis completed with issues\n")
                if results:
                    f.write(f"Error details: {results.get('error', 'Unknown error')}\n")
        
        if results and results['success']:
            print(f"\n🎊 SWEEP COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved:")
            print(f"   📄 Experiment guide: {results['experiment_guide_file']}")
            print(f"   🎬 Basic concatenated video: {results['video_file']}")
            if results.get('labeled_video_file'):
                print(f"   🏷️  Labeled concatenated video: {results['labeled_video_file']}")
            print(f"   🗺️  Video mapping: {results['video_mapping_file']}")
        else:
            print(f"\n⚠️  Sweep completed but analysis had issues. Check the output above.")
            
    except Exception as e:
        error_msg = f"Error during analysis: {e}"
        print(f"\n❌ {error_msg}")
        # Log analysis error
        with open(log_file, 'a') as f:
            f.write(f"\n❌ {error_msg}\n")
            f.write("You can manually run analysis later with:\n")
            f.write("python sweep_analyzer.py\n")
        print(f"   You can manually run analysis later with:")
        print(f"   python sweep_analyzer.py")

if __name__ == "__main__":
    main() 
    import time
    print("\n⏳ Waiting 5 seconds before suspend...")
    time.sleep(5)
    print("💤 Suspending computer...")
    subprocess.run(["systemctl", "shutdown"], check=True)