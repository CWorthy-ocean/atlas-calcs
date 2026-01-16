#!/usr/bin/env bash
#SBATCH --job-name atlas-calcs
#SBATCH --account m4632
#SBATCH --qos premium
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 02:00:00
#SBATCH --constraint cpu
#SBATCH --output logs/atlas-calcs-%j.out
#SBATCH --error logs/atlas-calcs-%j.err

set -euo pipefail

mkdir -p logs

module load python
conda activate atlas-calcs

python application.py parameters.yml