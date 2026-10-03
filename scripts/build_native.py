"""Build the optional parallel FSM kernel locally with the existing C++ compiler."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


root = Path(__file__).resolve().parents[1]
compiler = shutil.which("g++")
if not compiler:
    raise RuntimeError("Existing g++ is required for the optional OpenMP reference kernel")
source = root / "src/uav_mfg/fsm_kernel.cpp"
output = root / "build/fsm_kernel.so"
output.parent.mkdir(exist_ok=True)
command = [compiler, "-O3", "-std=c++17", "-fopenmp", "-shared", "-fPIC", str(source), "-o", str(output)]
subprocess.run(command, check=True)
metadata = {"command": command, "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
(output.parent / "fsm_build.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(output)
