# accept FR/EN aliment food

import pandas as pd
import ast
from typing import Optional, List

# Colonnes acceptées (FR/EN)
CANDIDATE_NAME_COLS = ["food", "aliment", "Aliment", "label", "name"]
CANDIDATE_GI_COLS   = ["gi", "IG", "ig"]
CANDIDATE_EMB_COLS  = ["embedding", "embeddings", "aliments_embedding"]

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive
    lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_gi_table(path: str = "glucy-pred/data/lookup/glycemic_index.csv") -> pd.DataFrame:
    """
    Charge un CSV d'IG avec ou sans embeddings.
    Colonnes possibles :
      - nom : food | Aliment | aliment | label | name
      - gi  : gi | IG | ig
      - embedding : embedding | embeddings | aliments_embedding (chaîne de type "[0.12, ...]")
    Retourne un DataFrame normalisé avec colonnes :
      - food (str), gi (float), embedding (list[float] ou None)
    """
    df = pd.read_csv(path)

    name_col = _find_col(df, CANDIDATE_NAME_COLS)
    gi_col   = _find_col(df, CANDIDATE_GI_COLS)
    emb_col  = _find_col(df, CANDIDATE_EMB_COLS)

    if name_col is None or gi_col is None:
        raise ValueError(
            f"CSV must have a name column ({CANDIDATE_NAME_COLS}) and a GI column ({CANDIDATE_GI_COLS}). "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["food"] = df[name_col].astype(str).str.strip()
    out["gi"]   = pd.to_numeric(df[gi_col], errors="coerce")

    if emb_col is not None:
        # la colonne est souvent une chaîne "[0.123, -0.456, ...]" -> on convertit en liste de floats
        def parse_vec(x):
            try:
                v = ast.literal_eval(x) if isinstance(x, str) else x
                if isinstance(v, (list, tuple)):
                    return [float(t) for t in v]
            except Exception:
                pass
            return None
        out["embedding"] = df[emb_col].apply(parse_vec)
    else:
        out["embedding"] = None

    # nettoyage de base
    out = out.dropna(subset=["food", "gi"]).reset_index(drop=True)
    return out

# Petit helper : récupération de l'IG par nom exact (insensible à la casse)
def get_gi_by_name(table: pd.DataFrame, name: str) -> Optional[float]:
    name_norm = str(name).strip().lower()
    m = table[table["food"].str.lower() == name_norm]
    if not m.empty:
        return float(m.iloc[0]["gi"])
    return None
