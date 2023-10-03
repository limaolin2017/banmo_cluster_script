#!/bin/bash
#SBATCH -J 3D_reconstruction      # 作业名称
#SBATCH -p medium               	     # Queue name
#SBATCH -N 1                      		     # 请求1个节点
#SBATCH --gres=gpu:5              	     # 请求5个GPU（或您需要的GPU数量）
#SBATCH --time=8:00:00                # 设置作业的最大运行时间为24小时

#SBATCH -o slurm_3D.%N.%J.out     # 标准输出文件的名称
#SBATCH -e slurm_3D.%N.%J.err      # 标准错误文件的名称

# 加载所需的模块（根据您的集群环境进行调整）
# module load python/3.8 CUDA/11.0 ...

# 路径到您的3D重建脚本所在的目录
cd /scripts/

# 运行您的3D重建脚本
gpus=5
seqname="cat-pikachiu"
addr=$3
use_human=$4
use_symm=$5
num_epochs=120
batch_size=256

savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --${use_symm}symm_shape \
  --${use_human}use_human

  
