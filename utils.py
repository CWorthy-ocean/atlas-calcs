"""
Utilities for launching a Dask cluster on NERSC.
"""

import os
import shutil
from subprocess import check_output, check_call

import tempfile
import time
import textwrap

import dask
from dask.distributed import Client, LocalCluster

import paths

JUPYTERHUB_URL = paths.jupyterhub_url

class dask_cluster(object):
    """Launch or connect to a Dask cluster on SLURM, or fall back to local."""

    def __init__(
        self,
        account=None,
        n_nodes=4,
        n_tasks_per_node=64,
        wallclock="04:00:00",
        queue_name="premium",
        scheduler_file=None,
    ):
        """
        Initialize a Dask cluster.

        Parameters
        ----------
        account : str, optional
            SLURM account to charge when launching a cluster.
        n_nodes : int, optional
            Number of SLURM nodes to request.
        n_tasks_per_node : int, optional
            Tasks per node for dask-worker.
        wallclock : str, optional
            Wall clock time for the SLURM job (HH:MM:SS).
        queue_name : str, optional
            SLURM QoS/queue name.
        scheduler_file : str or pathlib.Path, optional
            Existing scheduler file to connect to. If provided, skip launch.
        """
        self.scheduler_file = scheduler_file
        
        if not slurm_available():
            self.cluster = LocalCluster()
            self.client = Client(self.cluster)
            self.jobid = None
            self.dashboard_link = self.client.dashboard_link
            self.local_cluster = True
            print(f"Local cluster running at {self.client.dashboard_link}")
            return
        
        if self.scheduler_file is not None:
            self.scheduler_file = str(self.scheduler_file)
            if not os.path.exists(self.scheduler_file):
                raise FileNotFoundError(f"scheduler_file not found: {self.scheduler_file}")
            self.jobid = None
        
        else:
            if account is None:
                raise ValueError("account is required when not using a scheduler file")
            self.scheduler_file, self.jobid = self._launch_dask_cluster(
                account=account,
                n_nodes=n_nodes,
                n_tasks_per_node=n_tasks_per_node,
                wallclock=wallclock,
                queue_name=queue_name,
            )

        self.local_cluster = False
        dask.config.config["distributed"]["dashboard"]["link"] = self.dashboard_link
        dask.config.config["distributed"]["dashboard"][
            "link"
        ] = "{JUPYTERHUB_SERVICE_PREFIX}proxy/{host}:{port}/status"
        

        self.dashboard_link = f"{JUPYTERHUB_URL}/{self.client.dashboard_link}"
        self.client = Client(scheduler_file=self.scheduler_file)

        print(f"Dashboard:\n {self.dashboard_link}")

    def _launch_dask_cluster(self, account, n_nodes, n_tasks_per_node, wallclock, queue_name):
        """Submit a SLURM job that starts a Dask scheduler and workers."""
        path_dask = f"{paths.scratch}/dask"
        os.makedirs(path_dask, exist_ok=True)

        scheduler_file = tempfile.mktemp(
            prefix="dask_scheduler_file.", suffix=".json", dir=path_dask
        )

        script = textwrap.dedent(
            f"""\
            #!/bin/bash
            #SBATCH --job-name dask-worker
            #SBATCH --account {account}
            #SBATCH --qos={queue_name}
            #SBATCH --nodes={n_nodes}
            #SBATCH --ntasks-per-node={n_tasks_per_node}
            #SBATCH --time={wallclock}
            #SBATCH --constraint=cpu
            #SBATCH --error {path_dask}/dask-workers/dask-worker-%J.err
            #SBATCH --output {path_dask}/dask-workers/dask-worker-%J.out

            echo "Starting scheduler..."

            scheduler_file={scheduler_file}
            rm -f $scheduler_file

            module load python
            conda activate atlas-calcs

            #start scheduler
            DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
            DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
            dask scheduler \
                --interface hsn0 \
                --scheduler-file $scheduler_file &

            dask_pid=$!

            # Wait for the scheduler to start
            sleep 5
            until [ -f $scheduler_file ]
            do
                 sleep 5
            done

            echo "Starting workers"

            #start scheduler
            DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
            DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
            srun dask-worker \
            --scheduler-file $scheduler_file \
                --interface hsn0 \
                --nworkers 1 

            echo "Killing scheduler"
            kill -9 $dask_pid
            """
        )

        script_file = tempfile.mktemp(prefix="launch-dask.", dir=path_dask)
        with open(script_file, "w") as fid:
            fid.write(script)

        print(f"spinning up dask cluster with scheduler:\n  {scheduler_file}")
        jobid = (
            check_output(f"sbatch {script_file} " + "awk '{print $1}'", shell=True)
            .decode("utf-8")
            .strip()
            .split(" ")[-1]
        )

        while not os.path.exists(scheduler_file):
            time.sleep(5)

        return scheduler_file, jobid

    def shutdown(self):
        """Shutdown the Dask client and any launched cluster resources."""
        self.client.shutdown()
        if self.jobid:
            check_call(f"scancel {self.jobid}", shell=True)
        if getattr(self, "cluster", None) is not None:
            self.cluster.close()


def slurm_available() -> bool:
    """Return True if the SLURM scheduler command is available."""
    return shutil.which("sbatch") is not None
