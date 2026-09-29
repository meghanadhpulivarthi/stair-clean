import json

import pytest

from stair.core.report.train_report import (
    diagnose,
    find_latest_checkpoint_state,
    load_log_history,
    render_curves,
    split_log_history,
)


REPORT_CONFIG = {
    "overfitting_patience_evals": 3,
    "plateau_relative_improvement": 0.01,
    "plateau_patience_evals": 5,
    "lr_low_relative_improvement_threshold": 0.05,
    "lr_high_spike_relative_increase": 0.5,
}


def write_trainer_state(path, log_history):
    path.write_text(json.dumps({"log_history": log_history}))


def test_find_latest_checkpoint_state_picks_highest_numeric_step(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-10").mkdir(parents=True)
    (run_dir / "checkpoint-30").mkdir(parents=True)
    (run_dir / "checkpoint-best").mkdir(parents=True)
    write_trainer_state(run_dir / "checkpoint-10" / "trainer_state.json", [])
    write_trainer_state(run_dir / "checkpoint-30" / "trainer_state.json", [])

    latest_path = find_latest_checkpoint_state(str(run_dir))

    assert "checkpoint-30" in latest_path


def test_find_latest_checkpoint_state_raises_when_none_found(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        find_latest_checkpoint_state(str(run_dir))


def test_load_log_history_returns_the_list(tmp_path):
    state_path = tmp_path / "trainer_state.json"
    write_trainer_state(state_path, [{"loss": 1.0, "step": 1, "epoch": 0.1}])

    log_history = load_log_history(str(state_path))

    assert log_history == [{"loss": 1.0, "step": 1, "epoch": 0.1}]


def test_split_log_history_separates_train_and_eval_entries():
    log_history = [
        {"loss": 1.0, "step": 1, "epoch": 0.1},
        {"eval_loss": 0.9, "step": 1, "epoch": 0.1},
        {"loss": 0.8, "step": 2, "epoch": 0.2},
        {"eval_loss": 0.7, "step": 2, "epoch": 0.2},
    ]

    train_entries, eval_entries = split_log_history(log_history)

    assert len(train_entries) == 2
    assert len(eval_entries) == 2
    assert all("loss" in entry for entry in train_entries)
    assert all("eval_loss" in entry for entry in eval_entries)


def test_diagnose_with_no_eval_entries_says_so_and_skips_eval_diagnostics():
    diagnostics = diagnose(
        train_entries=[{"loss": 1.0, "step": 1, "epoch": 0.1}],
        eval_entries=[],
        report_config=REPORT_CONFIG,
        num_train_epochs=3,
    )

    joined_diagnostics = " ".join(diagnostics)
    assert "no eval data" in joined_diagnostics.lower()


def test_diagnose_with_one_eval_entry_says_not_enough_data():
    diagnostics = diagnose(
        train_entries=[{"loss": 1.0, "step": 1, "epoch": 0.1}],
        eval_entries=[{"eval_loss": 0.9, "step": 1, "epoch": 0.1}],
        report_config=REPORT_CONFIG,
        num_train_epochs=3,
    )

    joined_diagnostics = " ".join(diagnostics)
    assert "not enough" in joined_diagnostics.lower()


def test_diagnose_flags_overfitting_when_eval_loss_rises_while_train_loss_falls():
    # eval loss falls for the first 2 evals, then rises for 3 in a row
    # (index 3, 4, 5) — long enough to trip overfitting_patience_evals=3
    train_entries = [
        {"loss": 0.85, "step": 1, "epoch": 1},
        {"loss": 0.70, "step": 2, "epoch": 2},
        {"loss": 0.55, "step": 3, "epoch": 3},
        {"loss": 0.40, "step": 4, "epoch": 4},
        {"loss": 0.25, "step": 5, "epoch": 5},
        {"loss": 0.10, "step": 6, "epoch": 6},
    ]
    eval_entries = [
        {"eval_loss": 1.0, "step": 1, "epoch": 1},
        {"eval_loss": 0.8, "step": 2, "epoch": 2},
        {"eval_loss": 0.6, "step": 3, "epoch": 3},
        {"eval_loss": 0.7, "step": 4, "epoch": 4},
        {"eval_loss": 0.9, "step": 5, "epoch": 5},
        {"eval_loss": 1.1, "step": 6, "epoch": 6},
    ]

    diagnostics = diagnose(train_entries, eval_entries, REPORT_CONFIG, num_train_epochs=6)

    joined_diagnostics = " ".join(diagnostics).lower()
    assert "overfit" in joined_diagnostics


def test_diagnose_flags_plateau_when_eval_loss_stops_improving():
    train_entries = [{"loss": 1.0 - (0.05 * step), "step": step, "epoch": step} for step in range(1, 8)]
    eval_entries = [{"eval_loss": 0.5, "step": step, "epoch": step} for step in range(1, 8)]

    diagnostics = diagnose(train_entries, eval_entries, REPORT_CONFIG, num_train_epochs=7)

    joined_diagnostics = " ".join(diagnostics).lower()
    assert "plateau" in joined_diagnostics


def test_diagnose_flags_lr_too_low_when_train_loss_barely_moves():
    # train loss goes from 1.00 to 0.97 across the whole run: relative
    # improvement = (1.00 - 0.97) / 1.00 = 0.03, below the 0.05 threshold
    train_entries = [
        {"loss": 1.00, "step": 1, "epoch": 1},
        {"loss": 0.99, "step": 2, "epoch": 2},
        {"loss": 0.98, "step": 3, "epoch": 3},
        {"loss": 0.975, "step": 4, "epoch": 4},
        {"loss": 0.97, "step": 5, "epoch": 5},
    ]
    # eval loss keeps improving steadily so overfitting/plateau don't also fire
    eval_entries = [
        {"eval_loss": 1.0, "step": 1, "epoch": 1},
        {"eval_loss": 0.95, "step": 2, "epoch": 2},
        {"eval_loss": 0.90, "step": 3, "epoch": 3},
        {"eval_loss": 0.85, "step": 4, "epoch": 4},
        {"eval_loss": 0.80, "step": 5, "epoch": 5},
    ]

    diagnostics = diagnose(train_entries, eval_entries, REPORT_CONFIG, num_train_epochs=5)

    joined_diagnostics = " ".join(diagnostics).lower()
    assert "too low" in joined_diagnostics


def test_diagnose_flags_lr_too_high_when_train_loss_spikes_between_steps():
    # step 2 -> step 3 loss jumps from 0.90 to 1.50: relative increase =
    # (1.50 - 0.90) / 0.90 = 0.667, above the 0.5 threshold. Overall the run
    # still improves a lot (1.00 -> 0.40) so the "too low" diagnostic
    # shouldn't also fire.
    train_entries = [
        {"loss": 1.00, "step": 1, "epoch": 1},
        {"loss": 0.90, "step": 2, "epoch": 2},
        {"loss": 1.50, "step": 3, "epoch": 3},
        {"loss": 0.70, "step": 4, "epoch": 4},
        {"loss": 0.40, "step": 5, "epoch": 5},
    ]
    eval_entries = [
        {"eval_loss": 1.0, "step": 1, "epoch": 1},
        {"eval_loss": 0.9, "step": 2, "epoch": 2},
        {"eval_loss": 0.8, "step": 3, "epoch": 3},
        {"eval_loss": 0.7, "step": 4, "epoch": 4},
        {"eval_loss": 0.6, "step": 5, "epoch": 5},
    ]

    diagnostics = diagnose(train_entries, eval_entries, REPORT_CONFIG, num_train_epochs=5)

    joined_diagnostics = " ".join(diagnostics).lower()
    assert "too high" in joined_diagnostics
    assert "step 3" in joined_diagnostics


def test_render_curves_with_no_train_and_no_eval_entries_prints_placeholders(capsys):
    render_curves(train_entries=[], eval_entries=[])

    captured_output = capsys.readouterr().out
    assert "no training data" in captured_output.lower()
    assert "no eval data" in captured_output.lower()


def test_render_curves_with_train_entries_and_no_eval_entries_does_not_raise(capsys):
    train_entries = [
        {"loss": 1.0, "step": 1, "epoch": 1},
        {"loss": 0.8, "step": 2, "epoch": 2},
    ]

    render_curves(train_entries=train_entries, eval_entries=[])

    captured_output = capsys.readouterr().out
    assert "no eval data" in captured_output.lower()


def test_render_curves_with_eval_entries_and_no_train_entries_does_not_raise(capsys):
    eval_entries = [
        {"eval_loss": 0.9, "step": 1, "epoch": 1},
        {"eval_loss": 0.85, "step": 2, "epoch": 2},
    ]

    render_curves(train_entries=[], eval_entries=eval_entries)

    captured_output = capsys.readouterr().out
    assert "no training data" in captured_output.lower()


def test_render_curves_with_both_train_and_eval_entries_does_not_raise(capsys):
    train_entries = [
        {"loss": 1.0, "step": 1, "epoch": 1},
        {"loss": 0.8, "step": 2, "epoch": 2},
    ]
    eval_entries = [
        {"eval_loss": 0.9, "step": 1, "epoch": 1},
        {"eval_loss": 0.85, "step": 2, "epoch": 2},
    ]

    render_curves(train_entries=train_entries, eval_entries=eval_entries)

    captured_output = capsys.readouterr().out
    assert "no training data" not in captured_output.lower()
    assert "no eval data" not in captured_output.lower()
