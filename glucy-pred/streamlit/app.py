# =========================================================
# GluciPred - Streamlit app (from scratch, robust version)
# =========================================================
import base64, json
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# ------------------ CONFIG ------------------
st.set_page_config(page_title="GluciPred", page_icon="🍝", layout="wide")
DEFAULT_API = "https://data-jed-api-glucipred.hf.space/predict/image"

# State
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "annotated_bytes" not in st.session_state:
    st.session_state.annotated_bytes = None
if "diag" not in st.session_state:
    st.session_state.diag = {}

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    api_url = st.text_input("API endpoint", DEFAULT_API)
    total_weight = st.slider("Poids total estimé (g)", 150, 800, 350, 10)
    debug = st.checkbox("Mode debug", value=False)
    st.caption("Astuce : si 404, teste aussi /predict/image/ avec un slash final.")

# ------------------ HEADER ------------------
st.markdown(
    """
    <div style="text-align:center;margin:8px 0 16px 0">
      <h1>🍝 <span style="color:#4CAF50">GLUCY-Pred</span></h1>
      <p>Photo → segmentation → poids → glucides</p>
    </div>
    """,
    unsafe_allow_html=True
)
try:
    st.image("glucy_final_loop.gif", use_container_width=True)
except Exception:
    pass

# ------------------ HELPERS ------------------
def _b64(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def call_api(api_url: str, img_bytes: bytes, total_weight_g: int):
    """
    Essaie plusieurs formats de payload pour maximiser la compatibilité.
    """
    tries = [
        ("multipart:file", dict(files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                                data={"total_weight_g": total_weight_g})),
        ("multipart:image", dict(files={"image": ("image.jpg", img_bytes, "image/jpeg")},
                                 data={"total_weight_g": total_weight_g})),
        ("json:image_base64", dict(json={"image_base64": _b64(img_bytes),
                                         "total_weight_g": total_weight_g})),
        ("json:image", dict(json={"image": _b64(img_bytes),
                                  "total_weight_g": total_weight_g})),
        ("json:image_b64", dict(json={"image_b64": _b64(img_bytes),
                                      "total_weight_g": total_weight_g})),
    ]
    last = None
    for label, kwargs in tries:
        try:
            r = requests.post(api_url, timeout=60, **kwargs)
            if debug:
                st.write(f"⚙️ Try {label} → status {r.status_code}")
                st.write((r.text or "")[:1200])
            r.raise_for_status()
            js = r.json()
            # garde un aperçu brut dans Diagnostics
            st.session_state.diag["raw_json_preview"] = json.dumps(js)[:1500]
            return js
        except Exception as e:
            last = e
    raise last

def _get_first(js: dict, keys: list):
    """Retourne la première valeur trouvée parmi `keys`
       au niveau racine puis sous result/data."""
    for k in keys:
        if k in js:
            return js[k]
    for parent in ("result", "data", "output", "outputs"):
        d = js.get(parent)
        if isinstance(d, dict):
            for k in keys:
                if k in d:
                    return d[k]
    return None

def _walk_lists_of_dicts(obj):
    """Cherche récursivement des listes de dicts dans un JSON arbitraire."""
    found = []
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            found.append(obj)
        for x in obj:
            found.extend(_walk_lists_of_dicts(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_walk_lists_of_dicts(v))
    return found

def parse_response(js: dict):
    """
    Renvoie (annotated_bytes, df) où df = DataFrame brut AVANT enrichissement.
    """
    # image annotée (si fournie)
    b64 = _get_first(js, ["segmentated_image", "annotated_image", "image_b64", "image"])
    annotated_bytes = base64.b64decode(b64) if isinstance(b64, str) else None

    # 1) clés usuelles
    rows = _get_first(js, ["labels_and_weights", "items", "objects", "aliments", "predictions", "detections"])
    # 2) sinon fouille récursive
    if rows is None:
        candidates = _walk_lists_of_dicts(js)
        rows = candidates[0] if candidates else None

    # dataframe
    if isinstance(rows, list) and rows and isinstance(rows[0], str):
        df = pd.DataFrame({"label": rows})
    else:
        df = pd.DataFrame(rows or [])

    # mapping colonnes → noms standard
    rename_map = {
        # labels
        "class": "label", "name": "label", "label_name": "label", "class_name": "label",
        # confiance
        "conf": "confidence", "confidence_score": "confidence", "score": "confidence",
        # poids
        "poids": "weight_g", "poids_g": "weight_g", "weight": "weight_g",
        # % / ratio / aire
        "pourcentage": "percent_total", "pourcentage_total": "percent_total",
        "percent": "percent_total", "percentage": "percent_total",
        "area_percent": "percent_total",
        "ratio": "area_ratio", "proportion": "area_ratio",
        "mask_area": "area_pixels", "area": "area_pixels",
        # bboxes courantes
        "x1":"x1","y1":"y1","x2":"x2","y2":"y2","width":"width","height":"height","w":"width","h":"height",
        "bbox":"bbox"
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    st.session_state.diag["parsed_cols"] = list(df.columns)
    return annotated_bytes, df

# ---------- Nutrition loaders ----------
@st.cache_data
def load_ciqual(path="glucy-pred/data/lookup/ciqual_nutrition.csv"):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["food","carbs_per_100g"])
    name_col = next((c for c in ["food","aliment","Aliment","label","name"] if c in df.columns), None)
    carbs_col = next((c for c in ["carbs_per_100g","Glucides (g/100g)","glucides_100g","carbs"] if c in df.columns), None)
    if not name_col or not carbs_col:
        return pd.DataFrame(columns=["food","carbs_per_100g"])
    out = pd.DataFrame()
    out["food"] = df[name_col].astype(str).str.strip().str.lower()
    out["carbs_per_100g"] = pd.to_numeric(df[carbs_col], errors="coerce")
    return out.dropna()

@st.cache_data
def load_gi(path="glucy-pred/data/lookup/glycemic_index.csv"):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["food","gi"])
    name_col = next((c for c in ["food","aliment","Aliment","label","name"] if c in df.columns), None)
    gi_col = next((c for c in ["gi","IG","ig"] if c in df.columns), None)
    if not name_col or not gi_col:
        return pd.DataFrame(columns=["food","gi"])
    out = pd.DataFrame()
    out["food"] = df[name_col].astype(str).str.strip().str.lower()
    out["gi"] = pd.to_numeric(df[gi_col], errors="coerce")
    return out.dropna()

ALIASES = {
    # anglais -> français (quelques exemples utiles)
    "rice": "riz",
    "green beans": "haricot vert",
    "beans": "haricot",
    "broccoli": "brocoli",
    "cauliflower": "chou-fleur",
    "potato": "pomme de terre",
    "bread": "pain",
    "pasta": "pâtes",
    "tomato": "tomate",
    "carrot": "carotte",
    "chicken": "poulet",
    "beef": "boeuf",
    "pork": "porc",
    "fish": "poisson",
    "apple": "pomme",
    "banana": "banane",
    "strawberry": "fraise",
    "blueberry": "myrtille",
}

def _ensure_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tente de construire percent_total à partir de:
      - percent_total (déjà présent),
      - area_ratio (0..1 ou 0..100),
      - area_pixels (normalisation par la somme),
      - bboxes (x1,y1,x2,y2) ou (width,height) ou bbox [x,y,w,h].
    """
    if "percent_total" in df.columns:
        return df

    # ratio directement ?
    if "area_ratio" in df.columns:
        r = pd.to_numeric(df["area_ratio"], errors="coerce")
        if r.max(skipna=True) is not None:
            df["percent_total"] = r * (100.0 if r.max(skipna=True) <= 1.5 else 1.0)
            return df

    # aires de masques
    if "area_pixels" in df.columns:
        a = pd.to_numeric(df["area_pixels"], errors="coerce")
        s = a.sum(skipna=True)
        if s and s > 0:
            df["percent_total"] = 100.0 * a / s
            return df

    # bboxes (x1,y1,x2,y2)
    if {"x1","y1","x2","y2"}.issubset(df.columns):
        w = pd.to_numeric(df["x2"], errors="coerce") - pd.to_numeric(df["x1"], errors="coerce")
        h = pd.to_numeric(df["y2"], errors="coerce") - pd.to_numeric(df["y1"], errors="coerce")
        area = (w.clip(lower=0) * h.clip(lower=0)).fillna(0)
        s = area.sum()
        if s > 0:
            df["percent_total"] = 100.0 * area / s
            return df

    # bboxes (width,height)
    if {"width","height"}.issubset(df.columns):
        area = (pd.to_numeric(df["width"], errors="coerce").clip(lower=0) *
                pd.to_numeric(df["height"], errors="coerce").clip(lower=0)).fillna(0)
        s = area.sum()
        if s > 0:
            df["percent_total"] = 100.0 * area / s
            return df

    # bbox vectorisée [x,y,w,h]
    if "bbox" in df.columns:
        def _area_from_bbox(v):
            try:
                if isinstance(v, str):
                    v = json.loads(v)
                if isinstance(v, (list, tuple)) and len(v) >= 4:
                    return max(float(v[2]), 0.0) * max(float(v[3]), 0.0)
            except Exception:
                pass
            return 0.0
        area = df["bbox"].map(_area_from_bbox)
        s = area.sum()
        if s > 0:
            df["percent_total"] = 100.0 * area / s
            return df

    return df

def enrich_with_nutrition(df_in: pd.DataFrame, total_weight_g: int) -> pd.DataFrame:
    """Complète label/percent/weight_g puis calcule carbs_g (CIQUAL) et ajoute gi."""
    df = df_in.copy()

    # label
    if "label" not in df.columns:
        for c in ["name","class","class_name","label_name"]:
            if c in df.columns:
                df["label"] = df[c].astype(str)
                break

    # percent
    df = _ensure_percent(df)

    # poids (si pas fourni) à partir du % et du poids total
    if "weight_g" not in df.columns and "percent_total" in df.columns:
        perc = pd.to_numeric(df["percent_total"], errors="coerce")
        if perc.max(skipna=True) is not None and perc.max(skipna=True) <= 1.5:
            perc = perc * 100.0
        df["weight_g"] = total_weight_g * (perc / 100.0)

    # si toujours pas de poids → on ne peut pas calculer les glucides
    if "weight_g" not in df.columns:
        return df

    # clé d'appariement pour CIQUAL/IG
    if "label" in df.columns:
        df["__k"] = df["label"].astype(str).str.strip().str.lower()
        df["__k"] = df["__k"].map(lambda x: ALIASES.get(x, x))

    # CIQUAL → carbs
    ciqual = load_ciqual()
    if not ciqual.empty and "__k" in df.columns:
        df = df.merge(ciqual, left_on="__k", right_on="food", how="left").drop(columns=["food"])
        if "carbs_g" not in df.columns and "carbs_per_100g" in df.columns:
            df["carbs_g"] = df["weight_g"] * df["carbs_per_100g"] / 100.0

    # IG optionnel
    gi = load_gi()
    if not gi.empty and "__k" in df.columns and "gi" not in df.columns:
        df = df.merge(gi, left_on="__k", right_on="food", how="left").drop(columns=["food"])

    # ordre
    nice = [c for c in ["label","percent_total","weight_g","carbs_g","gi","confidence"] if c in df.columns]
    st.session_state.diag["enriched_cols"] = list(df.columns)
    return df[nice] if nice else df

# ------------------ UI ------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Importer une image")
    file = st.file_uploader("Choisir une image", type=["jpg","jpeg","png"])
    analyze = st.button("✅ Analyser") if file else None

    if file and analyze:
        img_bytes = file.read()
        with st.spinner("Appel API…"):
            try:
                js = call_api(api_url, img_bytes, total_weight)
                ann, df_parsed = parse_response(js)
                df_final = enrich_with_nutrition(df_parsed, total_weight)
                st.session_state.annotated_bytes = ann
                st.session_state.results_df = df_final
                st.session_state.diag["parsed_head"] = df_parsed.head(5).to_dict(orient="list")
                st.session_state.diag["final_head"] = df_final.head(5).to_dict(orient="list")
                st.success("Terminé ✅")
            except Exception as e:
                st.error(f"Erreur API : {e}")

    if file:
        st.caption("Original")
        st.image(Image.open(file), use_container_width=True)

with right:
    st.subheader("🧾 Résultats")
    df = st.session_state.results_df
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            if "weight_g" in df.columns:
                st.metric("Poids total estimé (g)", f"{df['weight_g'].sum():.0f}")
        with c2:
            if "carbs_g" in df.columns:
                st.metric("Glucides totaux (g)", f"{df['carbs_g'].sum():.1f}")
        st.download_button("📥 Télécharger CSV", df.to_csv(index=False).encode(), "results.csv", "text/csv")
    else:
        st.info("Charge une image puis clique sur Analyser.")

    if st.session_state.annotated_bytes:
        st.caption("Segmentation")
        st.image(Image.open(BytesIO(st.session_state.annotated_bytes)), use_container_width=True)

# ------------------ DIAGNOSTICS ------------------
with st.expander("🛠️ Diagnostics"):
    for k, v in st.session_state.diag.items():
        st.write(f"**{k}**:", v)


