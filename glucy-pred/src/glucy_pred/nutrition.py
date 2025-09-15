import pandas as pd
from .constants import NON_FOOD_CLASSES

def filter_food_rows(df: pd.DataFrame) -> pd.DataFrame:
    # suppose que df["label"] contient le label YOLO pour chaque instance
    return df[~df["label"].str.lower().isin({s.lower() for s in NON_FOOD_CLASSES})].copy()
