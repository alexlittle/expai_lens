import json
from pathlib import Path

class Logger:
    def __init__(self, log_dir="expai_lens_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def log_scalar(self, name, value, step):
        log = {"name": name, "value": value, "step": step, "type": "scalar"}
        with open(self.log_dir / "scalars.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")