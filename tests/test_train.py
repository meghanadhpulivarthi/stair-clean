import json
from pathlib import Path

import pytest

from stair.config import resolve_config
from stair.train import run_train


def make_complete_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"question": "q", "answer": "a", "reference": ["1"]}\n')
    (data_dir / "val.jsonl").write_text("")
    (data_dir / "test.jsonl").write_text("")
    toc = {"title": "Test Book", "table_of_contents": [{"id": "1", "section_num": "1", "title": "Intro", "leaf": True}]}
    (data_dir / "toc.json").write_text(json.dumps(toc))
    return data_dir


def make_fake_subprocess_run(returncode):
    calls = []

    def fake_subprocess_run(command, **kwargs):
        calls.append(command)

        class FakeCompletedProcess:
            pass

        completed_process = FakeCompletedProcess()
        completed_process.returncode = returncode
        return completed_process

    fake_subprocess_run.calls = calls
    return fake_subprocess_run


def test_run_train_raises_when_train_jsonl_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "val.jsonl").write_text("")
    (data_dir / "toc.json").write_text("{}")
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    with pytest.raises(FileNotFoundError, match="train.jsonl"):
        run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(0))


def test_run_train_raises_when_toc_json_missing(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("")
    (data_dir / "val.jsonl").write_text("")
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    with pytest.raises(FileNotFoundError, match="toc.json"):
        run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(0))


def test_run_train_invokes_torchrun_against_the_vendored_trainer(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)
    fake_subprocess_run = make_fake_subprocess_run(0)

    result = run_train(str(data_dir), str(run_dir), config, subprocess_run=fake_subprocess_run)

    assert result["returncode"] == 0
    assert len(fake_subprocess_run.calls) == 1
    invoked_command = fake_subprocess_run.calls[0]
    assert invoked_command[0] == "torchrun"
    assert "--nnodes=1" in invoked_command
    assert any("training_fsdp_trainer.py" in part for part in invoked_command)
    assert "--save_path" in invoked_command


def test_run_train_returns_nonzero_returncode_without_raising(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    result = run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(1))

    assert result["returncode"] == 1


def test_run_train_resolves_relative_data_and_run_dirs_to_absolute_paths(tmp_path, monkeypatch):
    # torchrun is launched with cwd=VENDORED_TRAINER_SCRIPT.parent, so a
    # relative --data/--out path must be resolved to an absolute path before
    # being handed to the CLI args builder, or the vendored trainer looks
    # for train.jsonl/writes checkpoints under the wrong directory
    data_dir = make_complete_data_dir(tmp_path)
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)
    fake_subprocess_run = make_fake_subprocess_run(0)

    monkeypatch.chdir(tmp_path)
    relative_data_dir = data_dir.relative_to(tmp_path)
    relative_run_dir = run_dir.relative_to(tmp_path)

    run_train(str(relative_data_dir), str(relative_run_dir), config, subprocess_run=fake_subprocess_run)

    invoked_command = fake_subprocess_run.calls[0]
    save_path_index = invoked_command.index("--save_path") + 1
    assert invoked_command[save_path_index] == str(run_dir.resolve())
    assert Path(invoked_command[save_path_index]).is_absolute()


def test_run_train_writes_a_valid_round_trippable_config_json(tmp_path):
    data_dir = make_complete_data_dir(tmp_path)
    run_dir = tmp_path / "run"
    config = resolve_config(override_path=None)

    run_train(str(data_dir), str(run_dir), config, subprocess_run=make_fake_subprocess_run(0))

    config_path = run_dir / "config.json"
    assert config_path.exists()
    with open(config_path, "r") as config_file:
        loaded_config = json.load(config_file)
    assert loaded_config == config
