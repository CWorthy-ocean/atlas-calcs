from pathlib import Path
from types import SimpleNamespace

import pytest

import application
import parsers


def test_load_yaml_params_single_document():
    yaml_path = Path(__file__).resolve().parents[1] / "cson_grids.yml"
    params = parsers.load_yaml_params(yaml_path)
    ccs = params["ccs"]

    assert ccs["nx"] == 224
    assert ccs["ny"] == 440
    assert ccs["size_x"] == 2688
    assert ccs["size_y"] == 5280
    assert ccs["center_lon"] == -134.5
    assert ccs["center_lat"] == 39.6
    assert ccs["rot"] == 33.3
    assert ccs["N"] == 100
    assert ccs["hc"] == 250
    assert ccs["theta_s"] == 6.0
    assert ccs["theta_b"] == 6.0
    assert ccs["verbose"] is True
    assert ccs["hmin"] == 5.0


def test_load_yaml_params_multiple_documents(tmp_path):
    yaml_path = tmp_path / "multi.yml"
    yaml_path.write_text(
        "---\nroms_tools_version: 3.3.0\n---\nGrid:\n  nx: 10\n",
        encoding="utf-8",
    )
    params = parsers.load_yaml_params(yaml_path)
    assert params["roms_tools_version"] == "3.3.0"
    assert params["Grid"]["nx"] == 10


def test_run_notebook_calls_papermill(monkeypatch, tmp_path):
    import nbformat

    calls = []

    def fake_execute_notebook(input_path, output_path, parameters, kernel_name=None):
        calls.append((input_path, output_path, parameters, kernel_name))

    monkeypatch.setitem(
        __import__("sys").modules,
        "papermill",
        SimpleNamespace(execute_notebook=fake_execute_notebook),
    )
    notebook_path = tmp_path / "a.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("No template")])
    nbformat.write(nb, str(notebook_path))
    output_dir = tmp_path / "executed"
    output_path = output_dir / "a.ipynb"
    params = {"nx": 10}

    application.run_notebook(
        notebook_path,
        output_path=output_path,
        parameters=params,
    )

    assert output_dir.exists()
    assert calls == [
        (str(notebook_path), str(output_path), params, "atlas-calcs"),
    ]


def test_run_notebook_requires_papermill(monkeypatch, tmp_path):
    import builtins
    import nbformat

    notebook_path = tmp_path / "a.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("No template")])
    nbformat.write(nb, str(notebook_path))

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "papermill":
            raise ImportError("papermill missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="papermill is required"):
        application.run_notebook(notebook_path, tmp_path / "out.ipynb", {})


def test_run_notebook_replaces_markdown_placeholders(monkeypatch, tmp_path):
    import nbformat

    notebook_path = tmp_path / "template.ipynb"
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Grid: {{ grid_yaml }}"),
            nbformat.v4.new_code_cell("print('ok')"),
        ]
    )
    nbformat.write(nb, str(notebook_path))
    output_path = tmp_path / "executed.ipynb"
    params = {"grid_yaml": "tests/_grid.yml"}

    captured = {}

    def fake_execute_notebook(input_path, output_path_arg, parameters, kernel_name=None):
        rendered = nbformat.read(input_path, as_version=4)
        captured["input_path"] = input_path
        captured["markdown"] = rendered.cells[0]["source"]

    monkeypatch.setitem(
        __import__("sys").modules,
        "papermill",
        SimpleNamespace(execute_notebook=fake_execute_notebook),
    )

    application.run_notebook(notebook_path, output_path, params)

    assert captured["input_path"] != str(notebook_path)
    assert captured["markdown"] == "Grid: tests/_grid.yml"


def test_normalize_file_type():
    assert parsers.normalize_file_type("roms-tools") == "roms-tools"
    assert parsers.normalize_file_type("roms_tools") == "roms-tools"
    with pytest.raises(ValueError, match="Supported file types"):
        parsers.normalize_file_type("other")


def test_load_roms_tools_object_grid_only(tmp_path):
    yaml_path = tmp_path / "grid.yml"
    yaml_path.write_text("---\nGrid:\n  nx: 10\n", encoding="utf-8")

    called = {}

    class FakeGrid:
        @staticmethod
        def from_yaml(path):
            called["path"] = path
            return "grid"

    module = SimpleNamespace(Grid=FakeGrid)
    result = parsers.load_roms_tools_object(yaml_path, roms_tools_module=module)

    assert result == "grid"
    assert called["path"] == str(yaml_path)


def test_load_roms_tools_object_other_class(tmp_path):
    yaml_path = tmp_path / "forcing.yml"
    yaml_path.write_text("---\nGrid:\n  nx: 10\nTidalForcing:\n  source: test\n", encoding="utf-8")

    called = {}

    class FakeTidal:
        @staticmethod
        def from_yaml(path):
            called["path"] = path
            return "forcing"

    module = SimpleNamespace(TidalForcing=FakeTidal, Grid=object())
    result = parsers.load_roms_tools_object(yaml_path, roms_tools_module=module)

    assert result == "forcing"
    assert called["path"] == str(yaml_path)


def test_load_roms_tools_object_requires_grid(tmp_path):
    yaml_path = tmp_path / "invalid.yml"
    yaml_path.write_text("---\nTidalForcing:\n  source: test\n", encoding="utf-8")
    module = SimpleNamespace(TidalForcing=object(), Grid=object())
    with pytest.raises(ValueError, match="must include a 'Grid' section"):
        parsers.load_roms_tools_object(yaml_path, roms_tools_module=module)


def test_load_roms_tools_object_multiple_sections(tmp_path):
    yaml_path = tmp_path / "invalid.yml"
    yaml_path.write_text(
        "---\nGrid:\n  nx: 10\nTidalForcing:\n  source: test\nSurfaceForcing:\n  source: test\n",
        encoding="utf-8",
    )
    module = SimpleNamespace(TidalForcing=object(), SurfaceForcing=object(), Grid=object())
    with pytest.raises(ValueError, match="only one non-Grid section"):
        parsers.load_roms_tools_object(yaml_path, roms_tools_module=module)


def test_dask_cluster_kwargs_model():
    model = parsers.DaskClusterKwargs(
        account="m4632",
        queue_name="premium",
        scheduler_file=None,
    )
    assert model.account == "m4632"
    assert model.queue_name == "premium"
    assert model.scheduler_file is None


def test_notebook_entry_model():
    config = parsers.NotebookConfig(
        parameters={"grid_yaml": "tests/_grid.yml", "test": True},
        output_path="executed/domain-sizing/example.ipynb",
    )
    entry = parsers.NotebookEntry(
        notebook_name="regional-domain-sizing",
        config=config,
    )
    assert entry.notebook_name == "regional-domain-sizing"
    assert entry.config.parameters["test"] is True


def test_notebook_list_model():
    config = parsers.NotebookConfig(
        parameters={"grid_yaml": "tests/_grid.yml"},
        output_path="executed/domain-sizing/example.ipynb",
    )
    entry = parsers.NotebookEntry(
        notebook_name="regional-domain-sizing",
        config=config,
    )
    notebook_list = parsers.NotebookList(title="Test", notebooks=[entry])
    assert notebook_list.notebooks[0].notebook_name == "regional-domain-sizing"


def test_parameters_config_model():
    config = parsers.NotebookConfig(
        parameters={"grid_yaml": "tests/_grid.yml"},
        output_path="executed/domain-sizing/example.ipynb",
    )
    entry = parsers.NotebookEntry(
        notebook_name="regional-domain-sizing",
        config=config,
    )
    notebook_list = parsers.NotebookList(title="Test", notebooks=[entry])
    dask_kwargs = parsers.DaskClusterKwargs(
        account="m4632",
        queue_name="premium",
        scheduler_file=None,
    )
    params = parsers.AppConfig(
        dask_cluster_kwargs=dask_kwargs,
        notebook_list=notebook_list,
    )
    assert params.dask_cluster_kwargs.account == "m4632"
    assert params.notebook_list.notebooks[0].notebook_name == "regional-domain-sizing"


def test_load_app_config(tmp_path):
    config_path = tmp_path / "parameters.yml"
    config_path.write_text(
        "\n".join(
            [
                "dask_cluster_kwargs:",
                "  account: m4632",
                "  queue_name: premium",
                "  scheduler_file: null",
                "",
                "notebooks:",
                "  title: Test",
                "  notebooks:",
                "- regional-domain-sizing:",
                "    parameters:",
                "      grid_yaml: tests/_grid.yml",
                "      test: true",
                "      scheduler_file: null",
                "    output_path: executed/domain-sizing/example.ipynb",
                "",
            ]
        ),
        encoding="utf-8",
    )

    app_config = parsers.load_app_config(config_path)

    assert app_config.dask_cluster_kwargs.account == "m4632"
    assert app_config.notebook_list.notebooks[0].notebook_name == "regional-domain-sizing"
    assert app_config.notebook_list.notebooks[0].config.parameters["test"] is True
    assert app_config.notebook_list.notebooks[0].config.parameters["grid_yaml"] == str(
        tmp_path / "tests/_grid.yml"
    )
    assert app_config.notebook_list.notebooks[0].config.output_path == str(
        tmp_path / "executed/domain-sizing/example.ipynb"
    )


def test_load_app_config_requires_notebooks(tmp_path):
    config_path = tmp_path / "parameters.yml"
    config_path.write_text(
        "\n".join(
            [
                "dask_cluster_kwargs:",
                "  account: m4632",
                "  queue_name: premium",
                "  scheduler_file: null",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="notebooks must be a list"):
        parsers.load_app_config(config_path)


def test_parse_notebook_entries_requires_list():
    with pytest.raises(ValueError, match="notebooks must be a list"):
        parsers._parse_notebook_entries({"title": "Bad"}, base_dir=Path("."))


def test_parse_notebook_entries_requires_single_key():
    with pytest.raises(ValueError, match="single-key mapping"):
        parsers._parse_notebook_entries([{"one": {}, "two": {}}], base_dir=Path("."))


def test_parse_notebook_entries_requires_mapping_payload():
    with pytest.raises(ValueError, match="payload must be a mapping"):
        parsers._parse_notebook_entries([{"name": "not-a-mapping"}], base_dir=Path("."))


def test_load_yaml_params_rejects_non_mapping(tmp_path):
    yaml_path = tmp_path / "bad.yml"
    yaml_path.write_text("---\n- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be mappings"):
        parsers.load_yaml_params(yaml_path)


def test_main_cli_uses_yaml_file(monkeypatch, tmp_path):
    captured = {}

    def fake_run_notebook(notebook_path, output_path, parameters, papermill_module=None):
        captured["notebook_path"] = notebook_path
        captured["output_path"] = output_path
        captured["parameters"] = parameters
        return None

    monkeypatch.setattr(application, "run_notebook", fake_run_notebook)

    config_path = tmp_path / "parameters.yml"
    config_path.write_text(
        "\n".join(
            [
                "notebooks:",
                "  title: Test",
                "  notebooks:",
                "  - regional-domain-sizing:",
                "      parameters:",
                "        grid_yaml: tests/_grid.yml",
                "        test: true",
                "      output_path: executed/domain-sizing/example.ipynb",
                "",
            ]
        ),
        encoding="utf-8",
    )
    args = [str(config_path)]

    application.main(args)

    assert captured["notebook_path"] == Path("regional-domain-sizing.ipynb")
    assert captured["output_path"] == tmp_path / "executed/domain-sizing/example.ipynb"
    assert captured["parameters"]["grid_yaml"] == str(tmp_path / "tests/_grid.yml")


def test_main_cli_requires_yaml_file():
    args = []
    with pytest.raises(SystemExit):
        application.main(args)


