import textwrap

import pytest

from stair.config import deep_merge, load_yaml_file, resolve_config


def test_load_yaml_file_reads_nested_structure(tmp_path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            model:
              name: some-model
              lora:
                rank: 16
            """
        )
    )
    loaded = load_yaml_file(config_path)
    assert loaded["model"]["name"] == "some-model"
    assert loaded["model"]["lora"]["rank"] == 16


def test_load_yaml_file_missing_path_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_file(missing_path)


def test_load_yaml_file_empty_file_returns_empty_dict(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    assert load_yaml_file(config_path) == {}


def test_deep_merge_overrides_nested_key_without_dropping_siblings():
    base = {
        "model": {"name": "base-model", "lora": {"rank": 8, "alpha": 16}},
        "training": {"epochs": 3},
    }
    override = {"model": {"lora": {"rank": 32}}}

    merged = deep_merge(base, override)

    assert merged["model"]["lora"]["rank"] == 32
    assert merged["model"]["lora"]["alpha"] == 16
    assert merged["model"]["name"] == "base-model"
    assert merged["training"]["epochs"] == 3


def test_deep_merge_does_not_mutate_inputs():
    base = {"model": {"lora": {"rank": 8}}}
    override = {"model": {"lora": {"rank": 32}}}

    deep_merge(base, override)

    assert base["model"]["lora"]["rank"] == 8
    assert override["model"]["lora"]["rank"] == 32


def test_resolve_config_with_no_override_returns_base_config():
    resolved = resolve_config(override_path=None)
    assert "model" in resolved
    assert "training" in resolved
    assert "data" in resolved


def test_resolve_config_applies_override(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        textwrap.dedent(
            """
            model:
              lora:
                rank: 64
            """
        )
    )
    resolved = resolve_config(override_path=override_path)
    assert resolved["model"]["lora"]["rank"] == 64


def test_resolve_config_empty_override_file_is_a_noop(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("# no changes yet\n")
    resolved = resolve_config(override_path=override_path)
    base_only = resolve_config(override_path=None)
    assert resolved == base_only


def test_load_yaml_file_non_mapping_document_raises_value_error(tmp_path):
    config_path = tmp_path / "list_document.yaml"
    config_path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml_file(config_path)


def test_resolve_config_rejects_unknown_override_key(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        textwrap.dedent(
            """
            model:
              lroa_rank: 64
            """
        )
    )
    with pytest.raises(ValueError, match="lroa_rank"):
        resolve_config(override_path=override_path)
