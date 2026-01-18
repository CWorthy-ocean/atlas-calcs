"""Helpers for parsing YAML and ROMS-Tools inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field



class NotebookConfig(BaseModel):
    """Schema for a notebook configuration."""

    parameters: Dict[str, Any]
    output_path: Union[Path, str]


class NotebookEntry(BaseModel):
    """Schema for a named notebook entry."""

    notebook_name: str
    config: NotebookConfig


class NotebookSection(BaseModel):
    """Schema for a titled notebook section."""

    title: str
    children: list[NotebookEntry]

    def to_toc_entry(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        children = []
        for entry in self.children:
            output_path = Path(entry.config.output_path)
            if base_dir is not None and output_path.is_absolute():
                try:
                    output_path = output_path.relative_to(base_dir)
                except ValueError:
                    pass
            children.append({"file": str(output_path)})
        return {"title": self.title, "children": children}


class NotebookList(BaseModel):
    """Schema for a list of notebook sections."""

    sections: list[NotebookSection]

    def iter_entries(self):
        for section in self.sections:
            for entry in section.children:
                yield entry

    def to_toc_entries(self, base_dir: Optional[Path] = None) -> list[Dict[str, Any]]:
        return [section.to_toc_entry(base_dir=base_dir) for section in self.sections]

class DaskClusterKwargs(BaseModel):
    """Schema for dask_cluster_kwargs configuration."""

    account: Optional[str] = None
    queue_name: Optional[str] = None
    n_nodes: Optional[int] = None
    n_tasks_per_node: Optional[int] = None
    wallclock: Optional[str] = None
    scheduler_file: Optional[str] = None


class AppConfig(BaseModel):
    """Top-level API schema for parameters.yml."""

    dask_cluster_kwargs: Optional[DaskClusterKwargs] = None
    notebook_list: NotebookList


def _parse_notebook_entry_list(raw_entries: list[Any], base_dir: Path) -> list[NotebookEntry]:
    entries = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError("Each notebook entry must be a single-key mapping.")
        if len(item) == 1:
            notebook_name, payload = next(iter(item.items()))
            if not isinstance(payload, dict):
                raise ValueError("Notebook entry payload must be a mapping.")
        else:
            notebook_keys = [key for key in item.keys() if isinstance(key, str) and key.endswith(".ipynb")]
            if len(notebook_keys) != 1:
                raise ValueError("Each notebook entry must be a single-key mapping.")
            notebook_name = notebook_keys[0]
            payload = {
                "parameters": item.get("parameters", {}),
                "output_path": item.get("output_path"),
            }
        parameters = dict(payload.get("parameters", {}))
        grid_yaml = parameters.get("grid_yaml")
        if isinstance(grid_yaml, str):
            grid_path = Path(grid_yaml)
            if not grid_path.is_absolute():
                parameters["grid_yaml"] = str(base_dir / grid_path)
        output_path = payload.get("output_path")
        if isinstance(output_path, str):
            output_path_value = Path(output_path)
            if not output_path_value.is_absolute():
                output_path = str(base_dir / output_path_value)
        config = NotebookConfig(
            parameters=parameters,
            output_path=output_path,
        )
        entries.append(NotebookEntry(notebook_name=notebook_name, config=config))
    return entries


def _parse_notebook_entries(raw_entries: Any, base_dir: Path) -> NotebookList:
    if isinstance(raw_entries, dict):
        title = raw_entries.get("title", "Untitled")
        children = raw_entries.get("children") or raw_entries.get("notebooks")
        if not isinstance(children, list):
            raise ValueError("children must be a list of entries.")
        sections = [NotebookSection(title=title, children=_parse_notebook_entry_list(children, base_dir))]
        return NotebookList(sections=sections)

    if isinstance(raw_entries, list):
        if raw_entries and all(isinstance(item, dict) and "children" in item for item in raw_entries):
            sections = []
            for section in raw_entries:
                title = section.get("title", "Untitled")
                children = section.get("children")
                if not isinstance(children, list):
                    raise ValueError("children must be a list of entries.")
                sections.append(NotebookSection(title=title, children=_parse_notebook_entry_list(children, base_dir)))
            return NotebookList(sections=sections)
        # Fall back to a single untitled section
        return NotebookList(
            sections=[
                NotebookSection(
                    title="Untitled",
                    children=_parse_notebook_entry_list(raw_entries, base_dir),
                )
            ]
        )

    raise ValueError("notebooks must be a list of sections.")


def load_yaml_params(path: Optional[Union[Path, str]]) -> Dict[str, Any]:
    """Load parameters from one or more YAML documents."""
    if path is None:
        return {}
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc]
    merged: Dict[str, Any] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            raise ValueError("YAML documents must be mappings.")
        merged.update(doc)
    return merged


def normalize_file_type(file_type: Optional[str]) -> Optional[str]:
    """Normalize a file type string."""
    if file_type is None:
        return None
    normalized = file_type.replace("_", "-").lower()
    if normalized not in {"roms-tools", "app-config"}:
        raise ValueError("Supported file types are 'roms-tools', 'roms_tools', or 'app-config'.")
    return normalized


def _select_roms_tools_class_name(yaml_params: Dict[str, Any]) -> str:
    """Determine the roms_tools class name to use based on YAML keys."""
    if "Grid" not in yaml_params:
        raise ValueError("ROMS-Tools YAML must include a 'Grid' section.")
    other_keys = sorted(
        key for key in yaml_params.keys() if key not in {"Grid", "roms_tools_version"}
    )
    if not other_keys:
        return "Grid"
    if len(other_keys) > 1:
        raise ValueError("ROMS-Tools YAML must include only one non-Grid section.")
    return other_keys[0]



def load_app_config(path: Union[Path, str]) -> AppConfig:
    """Load parameters.yml into an AppConfig object."""
    path_obj = Path(path)
    raw = load_yaml_params(path_obj)
    dask_kwargs = raw.get("dask_cluster_kwargs")
    if "notebooks" not in raw:
        raise ValueError("notebooks must be a list of entries.")
    notebooks_raw = raw.get("notebooks")
    notebook_list = _parse_notebook_entries(notebooks_raw, base_dir=path_obj.parent)
    return AppConfig(
        dask_cluster_kwargs=DaskClusterKwargs(**dask_kwargs) if dask_kwargs else None,
        notebook_list=notebook_list,
    )


def load_roms_tools_object(
    yaml_path: Union[Path, str],
    roms_tools_module: Any = None,
) -> Any:
    """Load a roms_tools object via its from_yaml method."""
    module = roms_tools_module
    if module is None:
        try:
            import roms_tools  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via explicit error path
            raise RuntimeError("roms_tools is required to load this YAML file.") from exc
        module = roms_tools

    yaml_path_obj = Path(yaml_path)
    yaml_params = load_yaml_params(yaml_path_obj)
    class_name = _select_roms_tools_class_name(yaml_params)

    if not hasattr(module, class_name):
        raise ValueError(f"roms_tools has no attribute '{class_name}'.")

    cls = getattr(module, class_name)

    if not hasattr(cls, "from_yaml"):
        raise ValueError(f"roms_tools.{class_name} has no from_yaml method.")
    return cls.from_yaml(str(yaml_path_obj))
