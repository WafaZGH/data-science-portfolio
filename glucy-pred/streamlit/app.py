import streamlit as st
from PIL import Image
from io import BytesIO
import base64, requests, pandas as pd

# --------- Config ---------
st.set_page_config(page_title="GluciPred", page_icon="🍝", layout="wide")

# Endpoint par défaut (modifiable dans la sidebar)
DEFAULT_API = "https://data-jed-api-glucipred.hf.space/predict/image"

# --------- State ---------
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "annotated_bytes" not in st.session_state:
    st.session_state.annotated_bytes = None

# --------- Sidebar ---------
with st.sidebar:
    st.header("⚙️ Paramètres")
    api_url = st.text_input("API endpoint", DEFAULT_API)
    total_weight = st.slider("Poids total estimé (g)", 150, 800, 350, 10)
    debug = st.checkbox("Mode debug", value=False)
    st.caption("Astuce : si /predict ne marche pas, essaie /predict/image (avec ou sans / final).")

# --------- Header ---------
st.markdown(
    """
    <div style="text-align:center;margin:6px 0 16px 0">
      <h1>🍝 <span style="color:#4CAF50">GLUCY-Pred</span></h1>
      <p>Photo → segmentation → poids & glucides</p>
    </div>
    """,
    unsafe_allow_html=True
)

# (Optionnel) GIF bannière si présent dans le dossier
try:
    st.image("glucy_final_loop.gif", use_container_width=True)
except Exception:
    pass

# --------- Helpers ---------
def _b64(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def call_api(api_url: str, img_bytes: bytes, total_weight_g: int):
    """
    Essaie plusieurs formats :
      1) multipart/form-data -> files={'file':(...)} + data
      2) multipart/form-data -> files={'image':(...)} + data
      3) JSON -> {'image_base64': '...'}
      4) JSON -> {'image': '...'}
      5) JSON -> {'image_b64': '...'}
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
                st.write((r.text or "")[:1000])
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
    raise last

def _get_first(js: dict, keys: list):
    """Retourne la première clé trouvée, au niveau racine puis sous result/data."""
    for k in keys:
        if k in js:
            return js[k]
    for parent in ("result", "data"):
        d = js.get(parent)
        if isinstance(d, dict):
            for k in keys:
                if k in d:
                    return d[k]
    return None

def parse_response(js: dict):
    """
    Normalise la réponse :
      - Image b64 : 'segmentated_image' | 'annotated_image' | 'image_b64' | 'image'
      - Items : 'labels_and_weights' | 'items' | 'objects' | 'aliments' | 'predictions' | 'detections'
    Retourne (annotated_bytes, df)
    """
    if debug:
        st.write("🔎 JSON keys:", list(js.keys()))

    # image annotée
    b64 = _get_first(js, ["segmentated_image", "annotated_image", "image_b64", "image"])
    annotated_bytes = base64.b64decode(b64) if isinstance(b64, str) else None

    # items
    rows = _get_first(js, ["labels_and_weights", "items", "objects", "aliments", "predictions", "detections"])

    # liste simple de strings ?
    if isinstance(rows, list) and rows and isinstance(rows[0], str):
        df = pd.DataFrame({"label": rows})
    else:
        df = pd.DataFrame(rows or [])

    # mapping colonnes → noms standard
    rename_map = {
        "poids": "weight_g",
        "poids_g": "weight_g",
        "weight": "weight_g",
        "pourcentage": "percent_total",
        "pourcentage_total": "percent_total",
        "percent": "percent_total",
        "percentage": "percent_total",
        "class": "label",
        "name": "label",
        "label_name": "label",
        "class_name": "label",
        "conf": "confidence",
        "confidence_score": "confidence",
        "score": "confidence",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # ordre agréable si colonnes présentes
    order = [c for c in ["label", "percent_total", "weight_g", "carbs_g", "confidence"] if c in df.columns]
    if order:
        df = df[order]
    return annotated_bytes, df

# --------- Nutrition loaders (cached) ---------
@st.cache_data
def load_ciqual(path="glucy-pred/data/lookup/ciqual_nutrition.csv"):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["food", "carbs_per_100g"])
    name_col = next((c for c in ["food","aliment","Aliment","label","name"] if c in df.columns), None)
    carbs_col = next((c for c in ["carbs_per_100g","Glucides (g/100g)","glucides_100g","carbs"] if c in df.columns), None)
    if not name_col or not carbs_col:
        return pd.DataFrame(columns=["food", "carbs_per_100g"])
    out = pd.DataFrame()
    out["food"] = df[name_col].astype(str).str.strip().str.lower()
    out["carbs_per_100g"] = pd.to_numeric(df[carbs_col], errors="coerce")
    return out.dropna()

@st.cache_data
def load_gi(path="glucy-pred/data/lookup/glycemic_index.csv"):
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["food", "gi"])
    name_col = next((c for c in ["food","aliment","Aliment","label","name"] if c in df.columns), None)
    gi_col = next((c for c in ["gi","IG","ig"] if c in df.columns), None)
    if not name_col or not gi_col:
        return pd.DataFrame(columns=["food", "gi"])
    out = pd.DataFrame()
    out["food"] = df[name_col].astype(str).str.strip().str.lower()
    out["gi"] = pd.to_numeric(df[gi_col], errors="coerce")
    return out.dropna()

def enrich_with_nutrition(df_in: pd.DataFrame, total_weight_g: int) -> pd.DataFrame:
    """Complète weight_g si absent, calcule carbs_g via CIQUAL, ajoute GI si dispo."""
    df = df_in.copy()

    # normaliser label
    if "label" not in df.columns:
        for c in ["name","class","class_name","label_name"]:
            if c in df.columns:
                df["label"] = df[c].astype(str)
                break

    # compléter weight_g
    if "weight_g" not in df.columns:
        # si percent_total existe → calculer un poids
        if "percent_total" in df.columns:
            perc = pd.to_numeric(df["percent_total"], errors="coerce")
            # si ce sont des ratios (<= 1.5), convertir en %
            if perc.max(skipna=True) is not None and perc.max(skipna=True) <= 1.5:
                perc = perc * 100.0
            df["weight_g"] = total_weight_g * (perc / 100.0)
        else:
            # tenter autres noms
            for c in ["poids","poids_g","weight"]:
                if c in df.columns:
                    df["weight_g"] = pd.to_numeric(df[c], errors="coerce")
                    break

    # si toujours pas de poids → on ne peut pas calculer les glucides
    if "weight_g" not in df.columns:
        return df

    # CIQUAL : carbs_per_100g
    ciqual = load_ciqual()
    if not ciqual.empty and "label" in df.columns:
        df["__k"] = df["label"].astype(str).str.strip().str.lower()
        df = df.merge(ciqual, left_on="__k", right_on="food", how="left").drop(columns=["__k","food"])
        if "carbs_g" not in df.columns and "carbs_per_100g" in df.columns:
            df["carbs_g"] = df["weight_g"] * df["carbs_per_100g"] / 100.0

    # IG optionnel
    gi = load_gi()
    if not gi.empty and "label" in df.columns and "gi" not in df.columns:
        df["__k"] = df["label"].astype(str).str.strip().str.lower()
        df = df.merge(gi, left_on="__k", right_on="food", how="left").drop(columns=["__k","food"])

    cols = [c for c in ["label","percent_total","weight_g","carbs_g","gi","confidence"] if c in df.columns]
    return df[cols] if cols else df

# --------- UI ---------
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Importer une image")
    file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
    analyze = st.button("✅ Analyser") if file else None

    if file and analyze:
        img_bytes = file.read()
        with st.spinner("Appel API…"):
            try:
                js = call_api(api_url, img_bytes, total_weight)
                ann, df = parse_response(js)
                # enrichissement local (poids / glucides / IG)
                df = enrich_with_nutrition(df, total_weight)
                st.session_state.annotated_bytes = ann
                st.session_state.results_df = df
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

