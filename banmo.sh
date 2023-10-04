#!/bin/bash
#SBATCH -J 3D_reconstruction      # name
#SBATCH -p medium               	     # Queue name
#SBATCH -N 1                      		     # 1 node
#SBATCH --gres=gpu:5              	     # GPUs per node
#SBATCH --time=8:00:00                # maximum execution time (HH:MM:SS)

#SBATCH -o slurm_3D.%N.%J.out     # Name of the standard output file
#SBATCH -e slurm_3D.%N.%J.err      # Name of the standard error file

# Load the required modules (adjust for your cluster environment)
# module load python/3.8 CUDA/11.0 ...

module load Miniconda3

# Activate your conda environment
source activate banmo

# Print some info about the assigned nodes
#echo "nvidia-smi:"
#nvidia-smi

#echo "nvcc --version:"
#nvcc --version


# Path to the directory where your 3D reconstruction script is located
cd ./banmo

# Run your 3D reconstruction script
gpus=5
seqname="cat-pikachiu"
addr="10001"
use_human="no"
use_symm="no"
pose_cnn_path="./banmo/mesh_material/posenet/quad.pth"
num_epochs=120
batch_size=256
model_prefix="3D_reconstruction_cat_pikachiu"

savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --${use_symm}symm_shape \
  --${use_human}use_human


