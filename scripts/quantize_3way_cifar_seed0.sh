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
# ===========================================
# change CKPT="${RUN_DIR}/ckpt.best.pth.tar" 
# to the last-epoch chkpt for fair comparison
# ===========================================
SEED=0
RUN_DIR="log/CLIP/cifar100/cifar100_CLIP-ViT-B/16_LA_None_exp_0.01_Focal-SAM_0.05_sched_none_seed_${SEED}_0_flat_gamma_5.5_sharpness_0.8"
CKPT="${RUN_DIR}/20_ckpt.pth.tar"
ARGS_FILE="${RUN_DIR}/args.txt"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint for seed ${SEED}: $CKPT"
  exit 1
fi
if [[ ! -f "$ARGS_FILE" ]]; then
  echo "Missing args_file for seed ${SEED}: $ARGS_FILE"
  exit 1
fi

for SCP in 'full' 'backbone_only' 'trainable'; do
echo "============================="
echo "GPTQ seed ${SEED}"
echo "============================="
python quantize_ptq.py \
  --model_dir "$CKPT" \
  --args_file "$ARGS_FILE" \
  --quant_method gptq \
  --format fp3 \
  --act_scheme none \
  --scope ${SCP} \
  --seed 0 \
  --output_dir "output_quant/cifar100_seed${SEED}_${SCP}"
done  
