#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Activate your conda environment (replace 'your_conda_env_name' with your actual environment name)
echo "Activating Conda environment: env_isaaclab"
source conda activate env_isaaclab
# Alternatively, if 'conda activate' is preferred and configured for your shell:
# conda activate your_conda_env_name

# Define the base command to avoid repetition
TRAIN_CMD="python scripts/rsl_rl/train.py --task=Loco --headless"
PLAY_CMD="python scripts/rsl_rl/play.py --task=Loco --headless --video --video_length 200 --enable_cameras"

# --- Parameter Set 1 ---
echo "Running Parameter Set 1..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=True \
    agent.algorithm.symmetry_cfg.use_mirror_loss=True \
    env.rewards.feet_slide.weight=-0.1

echo "Playing Parameter Set 1..."
$PLAY_CMD

# --- Parameter Set 2 ---
echo "Running Parameter Set 2..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=False \
    agent.algorithm.symmetry_cfg.use_mirror_loss=True \
    env.rewards.feet_slide.weight=-0.1

echo "Playing Parameter Set 2..."
$PLAY_CMD

# --- Parameter Set 3 ---
# Note: env.rewards.feet_slide.weight is not in your spec for set 3.
# If it should have a default or a specific value, add it.
echo "Running Parameter Set 3..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=False \
    agent.algorithm.symmetry_cfg.use_mirror_loss=True \
    agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.5

echo "Playing Parameter Set 3..."
$PLAY_CMD

# --- Parameter Set 4 ---
echo "Running Parameter Set 4..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=True \
    agent.algorithm.symmetry_cfg.use_mirror_loss=True \
    env.rewards.feet_slide.weight=-0.5

echo "Playing Parameter Set 4..."
$PLAY_CMD

# --- Parameter Set 5 ---
echo "Running Parameter Set 5..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=False \
    agent.algorithm.symmetry_cfg.use_mirror_loss=True \
    env.rewards.feet_slide.weight=-0.5

echo "Playing Parameter Set 5..."
$PLAY_CMD

# --- Parameter Set 6 ---
echo "Running Parameter Set 6..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=False \
    agent.algorithm.symmetry_cfg.use_mirror_loss=False \
    env.rewards.feet_slide.weight=-0.5

echo "Playing Parameter Set 6..."
$PLAY_CMD

# --- Parameter Set 7 ---
# Note: env.rewards.feet_slide.weight is not in your spec for set 7.
echo "Running Parameter Set 7..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=False \
    agent.algorithm.symmetry_cfg.use_mirror_loss=True \
    agent.algorithm.symmetry_cfg.mirror_loss_coeff=0.1

echo "Playing Parameter Set 7..."
$PLAY_CMD

# --- Parameter Set 8 ---
# Note: env.rewards.feet_slide.weight is not in your spec for set 8.
echo "Running Parameter Set 8..."
$TRAIN_CMD \
    agent.algorithm.symmetry_cfg.use_data_augmentation=False \
    agent.algorithm.symmetry_cfg.use_mirror_loss=False

echo "Playing Parameter Set 8..."
$PLAY_CMD

echo "All parameter sets finished."

# Deactivate Conda environment (optional)
# conda deactivate