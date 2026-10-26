#!/usr/bin/env python3
"""
Audits the raw dataset, saves descriptive statistics, missingness, and data dictionary.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from credit_xai.config import ExperimentConfig
from credit_xai.data import audit_dataset, get_data_dictionary
from credit_xai.utils import get_logger, save_json

logger = get_logger("audit_dataset")

def main():
    cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")
    raw_path = Path(cfg.data.raw_dir) / cfg.data.raw_filename
    
    if not raw_path.exists():
        logger.error(f"Raw dataset not found at {raw_path}. Run download_data.py first.")
        return

    df = pd.read_csv(raw_path)
    output_dir = Path("results/metadata")
    audit = audit_dataset(df, output_dir)
    
    # Save data dictionary
    dd = get_data_dictionary()
    save_json(dd, output_dir / "data_dictionary.json")
    
    logger.info("Dataset audit complete. Audit summary:")
    logger.info(f"  - Total records: {audit['total_rows']}")
    logger.info(f"  - Features count: {audit['total_columns']}")
    logger.info(f"  - Default class rate: {audit.get('default_rate_percent', 0):.2f}%")
    logger.info(f"  - Duplicate rows: {audit['duplicate_rows']}")

if __name__ == "__main__":
    main()
