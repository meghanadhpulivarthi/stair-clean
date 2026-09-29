import datetime
import json


def print_run_header(script_name, config):
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {script_name}")
    print(f"Config      : {json.dumps(config, indent=2)}")
    print("=" * 60)
