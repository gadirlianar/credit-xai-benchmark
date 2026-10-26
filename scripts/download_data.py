#!/usr/bin/env python3
"""
Downloads or synthesizes the Give Me Some Credit dataset and audits its integrity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from credit_xai.config import ExperimentConfig
from credit_xai.data import fetch_or_generate_dataset
from credit_xai.utils import get_logger

logger = get_logger("download_data")

def main():
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    output_path = Path(cfg.data.raw_dir) / cfg.data.raw_filename
    
    logger.info(f"Initiating dataset acquisition for {cfg.data.dataset_name}...")
    df = fetch_or_generate_dataset(
        output_path=output_path,
        subsample_size=cfg.data.subsample_size,
        seed=cfg.project.seed
    )
    logger.info(f"Dataset successfully acquired: {len(df)} rows, {len(df.columns)} columns.")

if __name__ == "__main__":
    main()
