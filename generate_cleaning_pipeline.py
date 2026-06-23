"""
Generate and save a fitted CleaningPipeline using synthetic data.

Run this once locally (after `git pull origin main`) to produce
models/cleaning_pipeline.pkl, then upload it to the GitHub Release.

Usage:
    python generate_cleaning_pipeline.py
"""

import joblib
from pathlib import Path

from src.ingestion.synthetic_data import generate_synthetic_990
from src.ingestion.cleaning_pipeline import (
    apply_exclusion_criteria,
    align_fiscal_years,
    CleaningPipeline,
)
from src.models.train import TRAIN_YEARS

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
OUT = MODELS_DIR / "cleaning_pipeline.pkl"

print("Generating synthetic panel (5,000 orgs)…")
df = generate_synthetic_990(n_orgs=5000, seed=42)
df = align_fiscal_years(apply_exclusion_criteria(df))

df_train = df[df["fiscal_year"].isin(TRAIN_YEARS)]
print(f"Training rows: {len(df_train):,}  ({df_train['fiscal_year'].min()}–{df_train['fiscal_year'].max()})")

cleaner = CleaningPipeline()
cleaner.fit(df_train)

joblib.dump(cleaner, OUT)
print(f"Saved → {OUT}")
