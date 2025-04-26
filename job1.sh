#!/bin/bash
#SBATCH --job-name=JupyterJob
#SBATCH --output=jupyter_%j.out
#SBATCH --error=jupyter_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu03
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4

# Move to job submission directory
cd $SLURM_SUBMIT_DIR

# Load your environment
source /home/u142201017/142201017/NLP/.venv/bin/activate

# Load CUDA module
# module purge
module load cuda/11.3

# Get the compute node information
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

# Get unused port number
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Print connection information
NODE=$(hostname)
LOGIN_NODE="192.168.1.133"
echo "Use the following command on your local machine:"
echo "ssh -L 8888:${NODE}:${PORT} $USER@${LOGIN_NODE}"

# Start JupyterLab with proper configuration
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --NotebookApp.token='' --NotebookApp.password=''
