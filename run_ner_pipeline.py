import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ROOT / "Ner-Extractor" / "entity_extractor.py",
    ROOT / "Ner-Extractor" / "relation_extractor.py",
]


def run_step(script: Path) -> None:
    print(f"Running {script.relative_to(ROOT)}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        for script in STEPS:
            run_step(script)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
