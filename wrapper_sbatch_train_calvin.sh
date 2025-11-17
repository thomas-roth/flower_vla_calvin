#!/bin/bash

if [[ ${1:-} == "debug" ]]; then
    SBATCH_SCRIPT="/home/hk-project-p0024638/uruox/code/iTRAP/iTRAP/models/flower_vla_calvin/sbatch_train_calvin_debug.sh"
else
    SBATCH_SCRIPT="/home/hk-project-p0024638/uruox/code/iTRAP/iTRAP/models/flower_vla_calvin/sbatch_train_calvin.sh"
fi

SLURM_LOG_DIR="/hkfs/work/workspace/scratch/uruox-itrap_flower_abc/logs/slurm/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
mkdir -p "$SLURM_LOG_DIR"

sbatch --output="${SLURM_LOG_DIR}/%x_%j.out" \
       --error="${SLURM_LOG_DIR}/%x_%j.err" \
       "$SBATCH_SCRIPT"
