#!/usr/bin/env python3
"""
Inspects and records hardware, operating system, and software dependencies.
"""
import sys
import platform
import os
import psutil
from pathlib import Path
from credit_xai.utils import save_json, get_logger

logger = get_logger("inspect_environment")

def inspect() -> dict:
    env_info = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
    }

    # Inspect library versions
    packages = [
        "numpy", "pandas", "scipy", "sklearn", "xgboost",
        "lightgbm", "shap", "lime", "torch", "matplotlib",
        "seaborn", "yaml", "pytest"
    ]
    lib_versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            lib_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            lib_versions[pkg] = "not_installed"

    env_info["libraries"] = lib_versions

    output_path = Path("results/metadata/environment_metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(env_info, output_path)

    logger.info("================ Environment Inspection ================")
    for k, v in env_info.items():
        if k != "libraries":
            logger.info(f"{k}: {v}")
    logger.info("Installed Libraries:")
    for pkg, ver in lib_versions.items():
        logger.info(f"  - {pkg}: {ver}")
    logger.info(f"Metadata saved to {output_path}")
    logger.info("=========================================================")
    return env_info

if __name__ == "__main__":
    inspect()
