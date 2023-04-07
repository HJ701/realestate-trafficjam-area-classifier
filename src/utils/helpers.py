from pathlib import Path
import json

def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def dump_json(obj, path):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))