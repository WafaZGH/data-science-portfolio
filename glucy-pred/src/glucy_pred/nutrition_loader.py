import pandas as pd
from typing import Optional, List

# Colonnes possibles (FR/EN) -> on normalise vers:
# label, carbs_per_100g, protein_per_100g, fat_per_100g, calories_kcal_per_100g
CANDIDATE_NAME  = [
    "label", "food", "aliment", "Aliment", "name",
    "alim_nom_eng", "alim_nom_fr", "alim_nom"
]

# Carbs totaux si dispo (sinon on recomposera)
CANDIDATE_CARBS = [
    "carbohydrates (g/100g)", "Carbohydrates (g/100g)",
    "carbohydrates_per_100g", "carbs_per_100g",
    "Glucides (g/100g)", "glucides (g/100g)", "glucides_g_100g",
    "carbs", "carbohydrates"
]

# Éléments pour recomposer carbs si besoin
CANDIDATE_SUGARS = ["Sugars (g/100g)", "sugars (g/100g)", "sugars_g_100g", "sugars"]
CANDIDATE_STARCH = ["Starch (g/100g)", "starch (g/100g)", "starch_g_100g", "starch"]
CANDIDATE_POLYOL = ["Polyols (g/100g)", "polyols (g/100g)", "polyols_g_100g", "polyols"]

CANDIDATE_PROT  = [
    "Proteins (g/100g)", "Protein (g/100g)", "protein_per_100g",
    "Protéines (g/100g)", "proteines_g_100g", "protein"
]
CANDIDATE_FAT   = [
    "Fat (g/100g)", "fat_per_100g",
    "Lipides (g/100g)", "lipides_g_100g", "fat"
]
CANDIDATE_KCAL  = [
    "Energy (kcal/100g)", "calories_kcal_per_100g",
    "Energie (kcal/100g)", "énergie (kcal/100g)", "kcal", "calories"
]

def _find(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _to_num(series: pd.Series) -> pd.Series:
    # gère 3,5 -> 3.5 et <0.2 -> 0.2
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace("<", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def load_ciqual(path: str = "glucy-pred/data/lookup/ciqual_nutrition.csv") -> pd.DataFrame:
    """
    Charge CIQUAL et renvoie un DF normalisé:
      label, carbs_per_100g, protein_per_100g, fat_per_100g, calories_kcal_per_100g
    Si 'carbs' n'existe pas, on tente carbs = sugars + starch + polyols.
    """
    df = pd.read_csv(path)

    name = _find(df, CANDIDATE_NAME)
    if name is None:
        raise ValueError(f"Missing name column. Tried: {CANDIDATE_NAME}. Found: {list(df.columns)}")

    carbs = _find(df, CANDIDATE_CARBS)
    sugars = _find(df, CANDIDATE_SUGARS)
    starch = _find(df, CANDIDATE_STARCH)
    polyol = _find(df, CANDIDATE_POLYOL)

    prot = _find(df, CANDIDATE_PROT)
    fat  = _find(df, CANDIDATE_FAT)
    kcal = _find(df, CANDIDATE_KCAL)

    out = pd.DataFrame()
    out["label"] = df[name].astype(str).str.strip()

    if carbs:
        out["carbs_per_100g"] = _to_num(df[carbs])
    else:
        # Recomposition si possible
        s = _to_num(df[sugars]) if sugars else 0
        st = _to_num(df[starch]) if starch else 0
        po = _to_num(df[polyol]) if polyol else 0
        out["carbs_per_100g"] = (s.fillna(0) + st.fillna(0) + po.fillna(0))

    out["protein_per_100g"] = _to_num(df[prot]) if prot else pd.NA
    out["fat_per_100g"]     = _to_num(df[fat])  if fat  else pd.NA
    out["calories_kcal_per_100g"] = _to_num(df[kcal]) if kcal else pd.NA

    return out.dropna(subset=["label"]).reset_index(drop=True)

def get_row_by_name(table: pd.DataFrame, name: str):
    m = table[table["label"].str.lower() == str(name).strip().lower()]
    return None if m.empty else m.iloc[0].to_dict()
