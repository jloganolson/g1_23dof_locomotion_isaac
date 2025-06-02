#!/usr/bin/env python3

import os
import subprocess
import re
from pathlib import Path
from datetime import datetime

def extract_parameters(experiment_dir):
    """Extract parameters from an experiment directory using grep."""
    agent_file = os.path.join(experiment_dir, "params", "agent.yaml")
    env_file = os.path.join(experiment_dir, "params", "env.yaml")
    
    params = {}
    
    # Extract agent parameters using grep
    if os.path.exists(agent_file):
        try:
            # Extract use_data_augmentation
            result = subprocess.run(['grep', 'use_data_augmentation:', agent_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                line = result.stdout.strip()
                params['use_data_augmentation'] = 'true' in line.lower()
            
            # Extract use_mirror_loss
            result = subprocess.run(['grep', 'use_mirror_loss:', agent_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                line = result.stdout.strip()
                params['use_mirror_loss'] = 'true' in line.lower()
            
            # Extract mirror_loss_coeff
            result = subprocess.run(['grep', 'mirror_loss_coeff:', agent_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                line = result.stdout.strip()
                match = re.search(r'mirror_loss_coeff:\s*([0-9.]+)', line)
                if match:
                    params['mirror_loss_coeff'] = float(match.group(1))
        except:
            pass
    
    # Extract environment parameters using grep
    if os.path.exists(env_file):
        try:
            # Get the feet_air_time section and its weight
            result = subprocess.run(['grep', '-A', '30', 'feet_air_time:', env_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout
                # Updated regex to handle indentation
                weight_match = re.search(r'^\s*weight:\s*([0-9.]+)', lines, re.MULTILINE)
                if weight_match:
                    params['feet_air_time_weight'] = float(weight_match.group(1))
        except:
            pass
    
    return params

def check_ffmpeg_capabilities():
    """Check what ffmpeg encoders and filters are available."""
    capabilities = {
        'has_libopenh264': False,
        'has_mpeg4': False,
        'has_drawtext': False
    }
    
    try:
        # Check encoders
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        if 'libopenh264' in result.stdout:
            capabilities['has_libopenh264'] = True
        if 'mpeg4' in result.stdout:
            capabilities['has_mpeg4'] = True
            
        # Check filters
        result = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True)
        if 'drawtext' in result.stdout:
            capabilities['has_drawtext'] = True
            
    except:
        pass
        
    return capabilities

def create_labeled_videos(experiments, timestamp, capabilities):
    """Create individual labeled videos for each experiment."""
    labeled_videos = []
    temp_dir = f"temp_labeled_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("🏷️  Creating labeled videos...")
    
    # Find a suitable font file
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "C:/Windows/Fonts/arial.ttf"       # Windows
    ]
    
    font_file = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_file = font_path
            break
    
    if not font_file:
        print(f"⚠️  No suitable font found, falling back to basic concatenation")
        return None, temp_dir
    
    print(f"   Using font: {font_file}")
    
    for i, exp in enumerate(experiments, 1):
        params = exp['params']
        
        # Create compact label text - avoid special characters for ffmpeg
        air_time = params.get('feet_air_time_weight', 'N/A')
        data_aug = 'T' if params.get('use_data_augmentation', False) else 'F'
        mirror_loss = 'T' if params.get('use_mirror_loss', False) else 'F'
        mirror_coeff = params.get('mirror_loss_coeff', 'N/A')
        
        # Use safe characters for ffmpeg filter (avoid : and = which are special)
        label_text = f"Exp{i:02d} Air{air_time} Aug{data_aug} Mirror{mirror_loss} Coeff{mirror_coeff}"
        output_video = os.path.join(temp_dir, f"labeled_{i:02d}.mp4")
        
        # Use mpeg4 encoder (more reliable than libopenh264)
        if capabilities['has_drawtext'] and capabilities['has_mpeg4']:
            cmd = [
                "ffmpeg", "-i", exp['video_path'],
                "-vf", f"drawtext=text='{label_text}':fontfile={font_file}:fontcolor=white:fontsize=18:x=10:y=10:box=1:boxcolor=black@0.8",
                "-c:v", "mpeg4", "-c:a", "copy",
                "-y", output_video
            ]
        else:
            print(f"⚠️  Required capabilities not available, falling back to basic concatenation")
            return None, temp_dir
        
        print(f"   Processing video {i}/{len(experiments)}: {label_text}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            labeled_videos.append(output_video)
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to process video {i}: {e.stderr[:100] if e.stderr else 'Unknown error'}")
            print(f"      Falling back to basic concatenation")
            return None, temp_dir
    
    return labeled_videos, temp_dir

def analyze_sweep_results(experiment_name, air_time_weights=None, use_data_augmentation=None, 
                         use_mirror_loss=None, mirror_loss_coeffs=None, base_logs_dir="logs/rsl_rl",
                         create_overlays=True):
    """
    Analyze sweep results and create concatenated video with parameter guide.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "g1_23dof_sweep_v4")
        air_time_weights (list): List of air time weights used in sweep
        use_data_augmentation (list): List of data augmentation values used
        use_mirror_loss (list): List of mirror loss values used  
        mirror_loss_coeffs (list): List of mirror loss coefficients used
        base_logs_dir (str): Base directory for logs
        create_overlays (bool): Whether to attempt creating text overlays on videos
    
    Returns:
        dict: Results summary with file paths and experiment count
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing sweep results for: {experiment_name}")
    print(f"{'='*60}")
    
    # Construct the sweep directory path
    sweep_dir = os.path.join(base_logs_dir, experiment_name)
    
    if not os.path.exists(sweep_dir):
        print(f"❌ Error: Sweep directory not found: {sweep_dir}")
        return None
    
    # Find all experiment directories
    experiment_dirs = sorted([d for d in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, d))])
    
    # Analyze all experiments
    experiments = []
    for exp_dir in experiment_dirs:
        full_path = os.path.join(sweep_dir, exp_dir)
        video_path = os.path.join(full_path, "videos", "play", "rl-video-step-0.mp4")
        
        if os.path.exists(video_path):
            params = extract_parameters(full_path)
            experiments.append({
                'directory': exp_dir,
                'video_path': video_path,
                'params': params
            })
    
    if not experiments:
        print(f"❌ No experiments with videos found in {sweep_dir}")
        return None
    
    print(f"📊 Found {len(experiments)} experiments with videos")
    
    # Create output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_guide_file = f"{experiment_name}_parameter_guide_{timestamp}.txt"
    video_mapping_file = f"{experiment_name}_video_mapping_{timestamp}.txt"
    concat_video_file = f"{experiment_name}_concatenated_{timestamp}.mp4"
    concat_labeled_video_file = f"{experiment_name}_concatenated_labeled_{timestamp}.mp4"
    
    # Create parameter guide
    with open(param_guide_file, "w") as f:
        f.write(f"Parameter Sweep Results: {experiment_name}\n")
        f.write("=" * (25 + len(experiment_name)) + "\n\n")
        f.write("Sweep configuration:\n")
        if air_time_weights:
            f.write(f"AIR_TIME_WEIGHTS = {air_time_weights}\n")
        if use_data_augmentation:
            f.write(f"USE_DATA_AUGMENTATION = {use_data_augmentation}\n")
        if use_mirror_loss:
            f.write(f"USE_MIRROR_LOSS = {use_mirror_loss}\n")
        if mirror_loss_coeffs:
            f.write(f"MIRROR_LOSS_COEFFS = {mirror_loss_coeffs}\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, exp in enumerate(experiments, 1):
            params = exp['params']
            f.write(f"Experiment {i:2d}: {exp['directory']}\n")
            f.write(f"  Air Time Weight:      {params.get('feet_air_time_weight', 'N/A')}\n")
            f.write(f"  Data Augmentation:    {params.get('use_data_augmentation', 'N/A')}\n")
            f.write(f"  Mirror Loss:          {params.get('use_mirror_loss', 'N/A')}\n")
            f.write(f"  Mirror Loss Coeff:    {params.get('mirror_loss_coeff', 'N/A')}\n")
            f.write("\n")
    
    print(f"✅ Parameter guide created: {param_guide_file}")
    
    # Check ffmpeg capabilities
    if create_overlays:
        capabilities = check_ffmpeg_capabilities()
        print(f"🔧 FFmpeg capabilities: drawtext={capabilities['has_drawtext']}, "
              f"libopenh264={capabilities['has_libopenh264']}, mpeg4={capabilities['has_mpeg4']}")
    else:
        capabilities = None
    
    # Try to create videos
    video_created = False
    labeled_video_created = False
    
    try:
        # Create basic concatenated video (always works)
        concat_file = f"temp_concat_{timestamp}.txt"
        with open(concat_file, "w") as f:
            for exp in experiments:
                f.write(f"file '{os.path.abspath(exp['video_path'])}'\n")
        
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-y", concat_video_file
        ]
        
        print("🎬 Creating basic concatenated video...")
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)
        print(f"✅ Basic concatenated video created: {concat_video_file}")
        video_created = True
        
        # Try to create labeled video if capabilities allow
        if create_overlays and capabilities and capabilities['has_drawtext']:
            labeled_videos, temp_dir = create_labeled_videos(experiments, timestamp, capabilities)
            
            if labeled_videos:
                # Create concat file for labeled videos
                labeled_concat_file = f"temp_labeled_concat_{timestamp}.txt"
                with open(labeled_concat_file, "w") as f:
                    for video in labeled_videos:
                        f.write(f"file '{os.path.abspath(video)}'\n")
                
                cmd = [
                    "ffmpeg", "-f", "concat", "-safe", "0", "-i", labeled_concat_file,
                    "-c", "copy", "-y", concat_labeled_video_file
                ]
                
                print("🏷️  Creating labeled concatenated video...")
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Cleanup
                os.remove(labeled_concat_file)
                import shutil
                shutil.rmtree(temp_dir)
                
                print(f"✅ Labeled concatenated video created: {concat_labeled_video_file}")
                labeled_video_created = True
            else:
                print(f"⚠️  Could not create labeled videos, only basic concatenation available")
        else:
            print(f"⚠️  Text overlays not supported, only basic concatenation created")
        
        # Create video mapping
        with open(video_mapping_file, "w") as f:
            f.write(f"Video Segment Mapping\n")
            f.write("=" * 21 + "\n\n")
            if labeled_video_created:
                f.write(f"Labeled video: {concat_labeled_video_file}\n")
            f.write(f"Basic video: {concat_video_file}\n\n")
            f.write("Each video segment corresponds to:\n")
            f.write(f"(Each segment is approximately 4 seconds long)\n\n")
            
            for i, exp in enumerate(experiments, 1):
                params = exp['params']
                air_time = params.get('feet_air_time_weight', 'N/A')
                data_aug = 'True' if params.get('use_data_augmentation', False) else 'False'
                mirror_loss = 'True' if params.get('use_mirror_loss', False) else 'False'
                mirror_coeff = params.get('mirror_loss_coeff', 'N/A')
                
                f.write(f"Segment {i:2d} ({(i-1)*4:2.0f}s-{i*4:2.0f}s): {exp['directory']}\n")
                f.write(f"    Air Time Weight:      {air_time}\n")
                f.write(f"    Data Augmentation:    {data_aug}\n")
                f.write(f"    Mirror Loss:          {mirror_loss}\n")
                f.write(f"    Mirror Loss Coeff:    {mirror_coeff}\n")
                f.write("\n")
        
        print(f"✅ Video mapping created: {video_mapping_file}")
        
    except Exception as e:
        print(f"⚠️  Video creation failed: {e}")
        print(f"   Parameter guide still available: {param_guide_file}")
    
    # Summary
    print(f"\n📋 SWEEP ANALYSIS COMPLETE:")
    print(f"   📊 Experiments analyzed: {len(experiments)}")
    print(f"   📄 Parameter guide: {param_guide_file}")
    if video_created:
        print(f"   🎬 Basic concatenated video: {concat_video_file}")
    if labeled_video_created:
        print(f"   🏷️  Labeled concatenated video: {concat_labeled_video_file}")
    if video_created or labeled_video_created:
        print(f"   🗺️  Video mapping: {video_mapping_file}")
    print(f"{'='*60}\n")
    
    return {
        'experiment_count': len(experiments),
        'param_guide_file': param_guide_file,
        'video_file': concat_video_file if video_created else None,
        'labeled_video_file': concat_labeled_video_file if labeled_video_created else None,
        'video_mapping_file': video_mapping_file if (video_created or labeled_video_created) else None,
        'success': video_created or labeled_video_created
    }

if __name__ == "__main__":
    # Example usage - can be called directly for testing
    analyze_sweep_results(
        experiment_name="g1_23dof_sweep_v4",
        air_time_weights=[1.0, 3.0, 5.0],
        use_data_augmentation=[False, True],
        use_mirror_loss=[False, True],
        mirror_loss_coeffs=[0.5, 1.0],
        create_overlays=True
    ) 