#!/usr/bin/env bash
# One-time environment setup. Run on a compute node, not the login node:
#   srun -p mit_normal -c 4 --mem=16G -t 01:00:00 bash setup_env.sh
set -eo pipefail

ENV=/orcd/scratch/orcd/006/usemil/envs/data_alloc

module load miniforge/25.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$ENV" python=3.11
conda activate "$ENV"

pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir -r requirements.txt

python -c "import torch, transformers, trl, peft, wandb; print('torch', torch.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| wandb', wandb.__version__)"
