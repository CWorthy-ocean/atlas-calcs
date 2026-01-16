#!/usr/bin/env bash
set -euo pipefail

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

python application.py parameters.yml