import json
import subprocess
from pathlib import Path

from stair.core.train.args_builder import build_training_cli_args
from stair.run_header import print_run_header


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_TRAINER_SCRIPT = REPO_ROOT / "src" / "stair" / "vendor" / "silt" / "training_fsdp_trainer.py"


def check_required_data_files(data_dir):
    data_dir_path = Path(data_dir)
    required_file_names = ["train.jsonl", "val.jsonl", "toc.json"]
    for required_file_name in required_file_names:
        required_file_path = data_dir_path / required_file_name
        if not required_file_path.exists():
            raise FileNotFoundError(
                f"Expected {required_file_name} under {data_dir} "
                f"(run `stair prepare-data` first to produce it)"
            )


def run_train(data_dir, run_dir, config, subprocess_run=None):
    if subprocess_run is None:
        subprocess_run = subprocess.run

    print_run_header("stair train", config)
    # the vendored trainer subprocess runs with cwd=VENDORED_TRAINER_SCRIPT.parent,
    # so a relative --data/--out path would resolve against the wrong directory
    # unless we resolve to absolute paths before doing anything else with them
    resolved_data_dir = str(Path(data_dir).resolve())
    resolved_run_dir = str(Path(run_dir).resolve())
    print(f"Training data directory: {resolved_data_dir}")
    print(f"Run directory: {resolved_run_dir}")
    print("Note: `stair train` requires a local CUDA GPU — there is no CPU fallback.")

    check_required_data_files(resolved_data_dir)

    Path(resolved_run_dir).mkdir(parents=True, exist_ok=True)

    config_path = Path(resolved_run_dir) / "config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
    print(f"Config saved: {config_path}")

    training_cli_args = build_training_cli_args(config, resolved_data_dir, resolved_run_dir)

    num_gpus = config["training"]["num_gpus"]
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=" + str(num_gpus),
        str(VENDORED_TRAINER_SCRIPT),
    ]
    command.extend(training_cli_args)

    print(f"Running: {' '.join(command)}")
    completed_process = subprocess_run(command, cwd=str(VENDORED_TRAINER_SCRIPT.parent))

    print(f"Training run finished with exit code {completed_process.returncode}")
    print(f"Checkpoints and training_config.json saved under: {resolved_run_dir}")

    return {"command": command, "returncode": completed_process.returncode}
