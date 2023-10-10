#!/bin/bash
#SBATCH -J 3D_reconstruction      # name
#SBATCH -p high              	     # Queue name
#SBATCH -N 2                      		     # 2 node
#SBATCH --gres=gpu:2              	     # GPUs per node
#SBATCH --exclude=node[001-018],node[031-032]  #排除节点
#SBATCH --time=99:99:99                # maximum execution time (HH:MM:SS)
#SBATCH -o slurm_3D.%N.%J.out       # Name of the standard output file
#SBATCH -e slurm_3D.%N.%J.err        # Name of the standard error file
#SBATCH --distribution=block

# Load the required modules (adjust for your cluster environment)

module load Miniconda3

cd ./banmo
eval "$(conda shell.bash hook)"

# Activate your conda environment
source activate banmo-cu113

# Run your 3D reconstruction script
gpus=4
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


