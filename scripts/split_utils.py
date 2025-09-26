import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def groups_by_day(df: pd.DataFrame, date_col: str):
    return pd.to_datetime(df[date_col]).dt.date

def split_train_val_test(df: pd.DataFrame, date_col: str, test_frac=0.20, val_frac=0.20, seed=42):
    groups = groups_by_day(df, date_col)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trval_idx, test_idx = next(gss1.split(df, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed+1)
    tr_idx, val_idx = next(gss2.split(df.iloc[trval_idx], groups=groups.iloc[trval_idx]))
    tr_idx = trval_idx[tr_idx]; val_idx = trval_idx[val_idx]
    return tr_idx, val_idx, test_idx

def grouped_kfold_indices(df: pd.DataFrame, date_col: str, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    groups = groups_by_day(df, date_col)
    return list(gkf.split(df, groups=groups))
