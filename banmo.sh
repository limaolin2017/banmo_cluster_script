#!/bin/bash
#SBATCH -J 3D_reconstruction      # name
#SBATCH -p high              	     # Queue name
#SBATCH -N 2                      		     # 2 node
#SBATCH --gres=gpu:2              	     # GPUs per node
#SBATCH --nodelist=node[019,020,021,022,023,024,025,026]    #指定优先使用节点
#SBATCH --time=99:99:99                # maximum execution time (HH:MM:SS)
#SBATCH -o slurm_3D.%N.%J.out     # Name of the standard output file
#SBATCH -e slurm_3D.%N.%J.err      # Name of the standard error file
#SBATCH --distribution=block



# Load the required modules (adjust for your cluster environment)
module load Miniconda3
module load foss/2020b
module load NCCL/2.9.9-CUDA-11.3.1
module load PyTorch/1.10.0-foss-2020b-CUDA-11.3.1
module load CUDA/11.3.1

# Activate your conda environment
conda activate banmo-cu113

# Path to the directory where your 3D reconstruction script is located
cd ./banmo

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


