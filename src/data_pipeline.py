"""
Data Pipeline Module for Project ACAT.
Handles OMOP clinical data transformations, comorbidity indexing,
and patient-level leakage-free stratified group partitioning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

FEATURE_COLUMNS = [
    'age', 'is_male', 'length_of_stay', 
    'ejection_fraction', 'serum_creatinine', 'systolic_bp', 
    'has_diabetes', 'prior_comorbidity_count'
]
TARGET_COLUMN = 'target_outcome_30d'

def split_cohort_by_patient(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """
    Splits cohort by patient groups using StratifiedGroupKFold to prevent data leakage.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET_COLUMN], groups=df['person_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    overlap = set(train_df['person_id']).intersection(set(test_df['person_id']))
    if len(overlap) > 0:
        raise ValueError(f"Data leakage detected! {len(overlap)} overlapping patient IDs.")
        
    return train_df, test_df
