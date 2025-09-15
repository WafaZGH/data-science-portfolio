import pandas as pd
from typing import List, Tuple

# On filtre les objets non-alimentaires (fork, plate, etc.) si le module est dispo
try:
    from .nutrition import filter_food_rows
except Exception:
    def filter_food_rows(df: pd.DataFrame) -> pd.DataFrame:
        return df

def enrichir_avec_poids(
    df_yolo: pd.DataFrame,
    total_weight_g: float = 350.0,
    filter_non_food: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Répartit un poids total estimé entre les items détectés en proportion de 'percent_total'.
    Attend un DataFrame avec au moins les colonnes: ['label', 'percent_total'].

    Retourne: (labels_list, df_out) où df_out contient une colonne 'poids' (en g).
    """
    if df_yolo is None or len(df_yolo) == 0:
        return [], pd.DataFrame(columns=["label", "percent_total", "poids"])

    df = df_yolo.copy()

    if filter_non_food:
        try:
            df = filter_food_rows(df)
        except Exception:
            pass

    if "percent_total" not in df.columns:
        raise ValueError("Le DataFrame doit contenir la colonne 'percent_total'.")

    total_pct = float(df["percent_total"].sum())
    if total_pct <= 0:
        df["poids"] = 0.0
        return df.get("label", pd.Series([], dtype=str)).tolist(), df

    df["poids"] = (total_weight_g * df["percent_total"] / total_pct).round(1)
    labels = df["label"].astype(str).tolist() if "label" in df.columns else []
    return labels, df
