"""Sequential three-step CUDA driver gates; no formal task is launched."""
from pathlib import Path
import subprocess
import sys

study = Path(__file__).resolve().parent
root = study.parents[1] / "outputs" / study.name
for variant in ("default", "reduced_space_weak", "space_time_weak"):
    print("CUDA three-step gate:", variant, flush=True)
    subprocess.run([sys.executable, str(study / "verify_driver.py"), "--device", "cuda", "--variant", variant,
                    "--output-root", str(root / ("driver-verification-" + variant))], check=True)
