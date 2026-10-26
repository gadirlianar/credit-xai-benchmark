#!/usr/bin/env python3
"""
Master End-to-End Pipeline Runner:
Executes environment inspection, data acquisition, dataset audit,
model training, smoke testing, benchmarking, and explanation evaluation.
"""
import sys
import subprocess
from pathlib import Path

scripts = [
    "scripts/inspect_environment.py",
    "scripts/download_data.py",
    "scripts/audit_dataset.py",
    "scripts/train_models.py",
    "scripts/run_smoke_tests.py",
    "scripts/run_benchmarks.py",
    "scripts/evaluate_explanations.py",
]

def main():
    print("================================================================================")
    print("STARTING FULL REPRODUCIBLE RESEARCH BENCHMARK PIPELINE")
    print("================================================================================")
    python_bin = sys.executable

    for script in scripts:
        print(f"\n>>> Running: {script}")
        res = subprocess.run([python_bin, script], env={"PYTHONPATH": "src"})
        if res.returncode != 0:
            print(f"FAILED: {script} exited with code {res.returncode}")
            sys.exit(res.returncode)
        print(f">>> Finished: {script} successfully.")

    print("\n================================================================================")
    print("ALL EXPERIMENTAL STAGES COMPLETED CLEANLY AND SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    main()
