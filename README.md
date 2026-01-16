# Ocean-CDR-Atlas-Calcs

Run a configured set of parameterized notebooks to generate atlas calculations and
export results. The workflow is driven by a single YAML file and a thin CLI wrapper.

## Quick Start

```bash
./run.sh parameters.yml
```

On SLURM systems:

```bash
./run.sh --sbatch parameters.yml
```

## `run.sh` Interface

`run.sh` is the primary entrypoint. It:

- Activates (or creates) the `atlas-calcs` conda environment.
- Installs the `atlas-calcs` Jupyter kernel if missing.
- Runs `application.py` with your YAML config.
- Optionally submits a SLURM job with `--sbatch`.

Usage:

```bash
./run.sh [--sbatch] <parameters.yml>
```

The `--sbatch` flag submits a short SLURM wrapper via a heredoc. Edit the SBATCH
directives in `run.sh` to tune wallclock, nodes, or CPU allocation.

## Configuration (`parameters.yml`)

The YAML file defines both cluster settings and the notebooks to execute.

```yaml
dask_cluster_kwargs:
  account: m4632
  queue_name: premium
  n_nodes: 1
  n_tasks_per_node: 128
  wallclock: 02:00:00
  scheduler_file: null

notebooks:
  title: "Domain Sizing"
  notebooks:
    - regional-domain-sizing:
        parameters:
          grid_yaml: cson_forge/blueprints/.../_grid.yml
          test: true
        output_path: executed/domain-sizing/example.ipynb
```

Notes:
- Relative paths resolve against the YAML file location.
- `output_path` is used in the MyST TOC and for notebook outputs.

## Dask Cluster Lifecycle

When SLURM is available, `application.py` uses `utils.dask_cluster` to manage a
single Dask cluster for the full run of notebooks:

The connected `scheduler_file` is injected into each notebook’s parameters as
`dask_cluster_kwargs.scheduler_file`, so all notebooks use the same cluster; 
the calls to `utils.dask_cluster` will **connect** to a scheduler rather 
than spin up a new cluster.

If SLURM is not available, the code falls back to a local Dask cluster.

## What `application.py` Does

- Loads `parameters.yml` into validated Pydantic models (`parsers.py`).
- Optionally launches or connects to a Dask cluster (`utils.dask_cluster`).
- Executes each notebook with Papermill using the `atlas-calcs` kernel.
- Renders `{{ key }}` placeholders in markdown cells using the parameters dict.
- Updates `myst.yml` `project.toc` with the executed notebook paths.
- Logs failures and raises a summary error if any notebooks fail.

## MyST TOC Updates

After execution, `myst.yml` is updated to the layout:

```yaml
project:
  toc:
    - file: README.md
    - title: Domain Sizing
      children:
        - file: executed/domain-sizing/example.ipynb
```

## Development Notes

- Python dependencies live in `environment.yml`.
- Tests are under `tests/` (run with `pytest`).
- Parsing logic lives in `parsers.py`.
- Dask SLURM cluster management lives in `utils.py`.

