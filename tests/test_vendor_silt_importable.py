import subprocess
import sys
from pathlib import Path


VENDOR_SILT_DIR = Path(__file__).resolve().parents[1] / "src" / "stair" / "vendor" / "silt"


def test_vendored_silt_modules_import_without_error():
    import_check_script = (
        "import arguments\n"
        "import constants\n"
        "import trainers\n"
        "import train_utils\n"
        "print('IMPORTS_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", import_check_script],
        cwd=str(VENDOR_SILT_DIR),
        capture_output=True,
        text=True,
    )
    assert "IMPORTS_OK" in result.stdout, result.stderr


def test_vendored_training_script_shows_no_hardcoded_hf_token():
    training_script_path = VENDOR_SILT_DIR / "training_fsdp_trainer.py"
    training_script_text = training_script_path.read_text()
    assert "HF_TOKEN" not in training_script_text


def test_vendored_arguments_module_exposes_get_args():
    import_check_script = (
        "import arguments\n"
        "assert hasattr(arguments, 'get_args')\n"
        "assert hasattr(arguments, 'TrainingArgs')\n"
        "print('INTERFACE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", import_check_script],
        cwd=str(VENDOR_SILT_DIR),
        capture_output=True,
        text=True,
    )
    assert "INTERFACE_OK" in result.stdout, result.stderr
