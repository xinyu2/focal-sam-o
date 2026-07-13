#!/bin/bash
#SBATCH --partition=camas  ### Partition
#SBATCH --job-name=natbase  ### Job Name
#SBATCH --time=40:00:00      ### WallTime
#SBATCH --nodes=1            ### Number of Nodes
#SBATCH --ntasks-per-node=16 ### Number of tasks (MPI processes)
#SBATCH --gres=gpu:tesla:1    ### number of GPUs
#SBATCH --mem=300000 	### Memory(MB)

set -euo pipefail

module purge

FOCAL_SAM_ROOT=/weka/data/lab/yan/xinyu/ProPose/Focal-SAM-o
cd "$FOCAL_SAM_ROOT"
source /weka/data/lab/yan/xinyu/ProPose/Focal-SAM/.venv/bin/activate

echo "Running from: $(pwd)"
echo "Python: $(which python)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

for SEED in 0 1 42 123 777 1234; do
echo "============================="
echo "CLIP CIFAR-100 seed ${SEED}"
echo "============================="
python cifar_train_sam_clip.py --gpu 0 \
    --imb_factor 0.01 --loss_type LA --SAM_type Focal-SAM --rho 0.05 \
    --dataset cifar100 --seed "${SEED}" --flat_gamma 5.5 --sharpness 0.8 --arch CLIP-ViT-B/16 \
    --root_log "./log/CLIP/cifar100" --root_model "./log/CLIP/cifar100" \
    --epochs 20 --adaptformer --lr 0.01 --wd 5e-4
done
