#!/usr/bin/env bash
set -euo pipefail

sbatch_flag=false
if [[ ${1:-} == "--sbatch" ]]; then
  sbatch_flag=true
  shift
fi

if [[ -z ${1:-} ]]; then
  echo "Usage: $0 [--sbatch] <parameters.yml>"
  exit 1
fi
if [[ -n ${2:-} ]]; then
  echo "Usage: $0 [--sbatch] <parameters.yml>"
  exit 1
fi

yaml_file=$1

if $sbatch_flag; then
  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name atlas-calcs
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --output atlas-calcs-%j.out
#SBATCH --error atlas-calcs-%j.err
#SBATCH --time 02:00:00

set -euo pipefail
cd /Users/mclong/codes/atlas-calcs
./run.sh "$yaml_file"
EOF
  exit $?
fi

if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load conda
  fi
fi

if command -v conda >/dev/null 2>&1; then
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -q '^atlas-calcs$'; then
    conda env create -f environment.yml
  fi
  conda activate atlas-calcs
  set -u
else
  echo "Conda is not installed. Please install it and try again."
  exit 1
fi

if ! python -m jupyter kernelspec list 2>/dev/null | grep -q "atlas-calcs"; then
  python -m ipykernel install --user --name atlas-calcs --display-name "atlas-calcs"
fi

python application.py "$yaml_file"