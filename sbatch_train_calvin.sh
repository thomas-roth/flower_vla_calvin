#!/bin/bash

# Job Settings
#SBATCH -A hk-project-p0024638  # Project name
#SBATCH -J iTRAP_FLOWER_ABC     # Job name

# Cluster Settings
#SBATCH -p accelerated          # Partition name
#SBATCH -n 4                    # Number of tasks
#SBATCH --ntasks-per-node=4     # Number of tasks per node
#SBATCH --gres=gpu:4            # Number of GPUs
#SBATCH -c 4                    # Number of cores per task
#SBATCH -t 12:00:00 ## 12:00:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00

# Define the paths for storing output and error files
#SBATCH --output=/hkfs/work/workspace/scratch/uruox-itrap_flower_abc/logs/slurm/%x_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/uruox-itrap_flower_abc/logs/slurm/%x_%j.err

# -------------------------------

# Create necessary dirs in tmpdir
mkdir -p $TMPDIR/data $TMPDIR/logs

# Load dataset into tmpdir & log progress to stdout
(pv -f -w 90 -N "Extracting dataset" /hkfs/work/workspace/scratch/uruox-itrap_flower_abc/data/task_ABC_D.tar | tar -C $TMPDIR/data -xf -) 2>&1
(pv -f -w 90 -N "Extracting dataset annotations" /home/hk-project-p0024638/uruox/data/iTRAP-FLOWER/calvin_policy_dataset/vis_lang_clip_resnet50.tar | tar -C $TMPDIR/data -xf -) 2>&1

# Activate the virtualenv / conda environment
source /home/hk-project-p0024638/uruox/miniconda3/bin/activate itrap

export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

srun python /home/hk-project-p0024638/uruox/code/iTRAP/iTRAP/models/flower_vla_calvin/flower/training_calvin.py seed=42
# batch_size=2 model=smolflow_agent # batch_size=4 model.use_lora=False # model=vlm_berg_agent # batch_size=2 model.vla_mode='reduced_head' #model.use_perceiver=False model.use_incontext=True #seed=242 #model=mode_agent

# Sync logs of slurm & run to workspace
rsync -av $TMPDIR/logs /hkfs/work/workspace/scratch/uruox-itrap_flower_abc/logs
