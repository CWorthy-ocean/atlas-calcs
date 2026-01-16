import os
from pathlib import Path

scratch = Path(os.environ.get("SCRATCH", Path(os.environ.get("HOME")) / "scratch"))
jupyterhub_url = "https://jupyter.nersc.gov"