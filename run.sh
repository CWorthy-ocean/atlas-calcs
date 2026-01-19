#!/usr/bin/env bash
set -euo pipefail

sbatch_flag=false
test_flag=false
while [[ ${1:-} == --* ]]; do
  case "$1" in
    --sbatch)
      sbatch_flag=true
      ;;
    --test)
      test_flag=true
      ;;
    *)
      echo "Usage: $0 [--sbatch] [--test] <parameters.yml>"
      exit 1
      ;;
  esac
  shift
done

if [[ -z ${1:-} ]]; then
  echo "Usage: $0 [--sbatch] [--test] <parameters.yml>"
  exit 1
fi
if [[ -n ${2:-} ]]; then
  echo "Usage: $0 [--sbatch] [--test] <parameters.yml>"
  exit 1
fi

yaml_file=$1
test_arg=""
if $test_flag; then
  test_arg="--test"
fi

if $sbatch_flag; then
  submit_dir="$PWD"
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
cd "$submit_dir"
./run.sh $test_arg "$yaml_file"
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

echo "python application.py $test_arg $yaml_file"