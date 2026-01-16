"""Run parameterized notebooks with Papermill using YAML/JSON inputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, Iterable, Optional, Union

try:
    import papermill
except ImportError:  # pragma: no cover - exercised via explicit error path
    papermill = None

from parsers import load_app_config
import utils


logger = logging.getLogger(__name__)

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _render_markdown_placeholders(
    notebook_path: Path, parameters: Dict[str, Any]
) -> Optional[Path]:
    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)
    updated = False
    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if not source:
            continue

        def _replace(match):
            key = match.group(1)
            if key not in parameters:
                return match.group(0)
            return str(parameters[key])

        new_source = _PLACEHOLDER_PATTERN.sub(_replace, source)
        if new_source != source:
            cell["source"] = new_source
            updated = True

    if not updated:
        return None

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ipynb",
            delete=False,
            encoding="utf-8",
        ) as handle:
            nbformat.write(nb, handle)
            temp_path = handle.name
    except Exception:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
        raise
    return Path(temp_path)


def _update_myst_toc(myst_path: Path, toc_entry: Any) -> None:
    import yaml

    data = yaml.safe_load(myst_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("myst.yml must contain a top-level mapping.")
    project = data.get("project")
    if not isinstance(project, dict):
        project = {}
        data["project"] = project
    project["toc"] = toc_entry
    myst_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_notebook(
    notebook_path: Path,
    output_path: Path,
    parameters: Dict[str, Any],
    kernel_name: str = "atlas-calcs",
) -> None:
    """Execute notebooks with papermill and return output paths."""
    try:
        import papermill
    except ImportError:  # pragma: no cover - exercised via explicit error path
        raise RuntimeError("papermill is required to execute notebooks.")
    module = papermill

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _render_markdown_placeholders(notebook_path, parameters)
    input_path = temp_path if temp_path is not None else notebook_path
    try:
        return module.execute_notebook(
            str(input_path),
            str(output_path),
            parameters=parameters,
            kernel_name=kernel_name,
        )
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run parameterized notebooks with papermill.")
    parser.add_argument(
        "yaml_file",
        help="Path to parameters.yml file.",
    )
    return parser.parse_args(args=args)


def main(args: Optional[Iterable[str]] = None) -> int:
    """CLI entrypoint for running notebooks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parsed = parse_args(args=args)
    app_config = load_app_config(Path(parsed.yaml_file))
    dask_cluster_kwargs = (
        app_config.dask_cluster_kwargs.model_dump()
        if app_config.dask_cluster_kwargs is not None
        else None
    )

    if utils.slurm_available():
        cluster = utils.dask_cluster(**dask_cluster_kwargs) if dask_cluster_kwargs else None
    else:
        cluster = None
    
    completed = []
    failed = []
    try:
        for entry in app_config.notebook_list.notebooks:
            parameters = dict(entry.config.parameters)
            
            if dask_cluster_kwargs is not None:
                parameters["dask_cluster_kwargs"] = dask_cluster_kwargs
            if cluster is not None:
                parameters["dask_cluster_kwargs"]["scheduler_file"] = cluster.scheduler_file
            
            notebook_path = Path(entry.notebook_name)
            if notebook_path.suffix == "":
                notebook_path = notebook_path.with_suffix(".ipynb")
            output_path = Path(entry.config.output_path)
            if output_path.suffix == "":
                output_path = output_path.with_suffix(".ipynb")
            
            logger.info("Running %s -> %s", notebook_path, output_path)
            try:
                run_notebook(
                    notebook_path,
                    output_path=output_path,
                    parameters=parameters,
                )
                completed.append(str(output_path))
                logger.info("Completed %s", output_path)
            except Exception as exc:
                failed.append(str(output_path))
                logger.exception("Failed %s: %s", output_path, exc)
    finally:
        if cluster is not None:
            logger.info("Shutting down cluster")
            cluster.shutdown()
    
    myst_path = Path("myst.yml")
    toc_entry = app_config.notebook_list.to_toc_entry(base_dir=Path(parsed.yaml_file).parent)
    toc_list = [{"file": "README.md"}, toc_entry]
    try:
        if myst_path.exists():
            _update_myst_toc(myst_path, toc_list)
            logger.info("Updated myst.yml toc at %s", myst_path)
        else:
            logger.warning("myst.yml not found at %s; skipping toc update", myst_path)
    except Exception:
        logger.exception("Failed to update myst.yml toc at %s", myst_path)
    if failed:
        raise RuntimeError(
            "Notebook execution failures. Completed: {completed}; Failed: {failed}".format(
                completed=completed, failed=failed
            )
        )
    return 0

if __name__ == "__main__":
    sys.exit(main())
