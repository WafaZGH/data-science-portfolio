import streamlit as st
from PIL import Image
from io import BytesIO
import base64, requests, pandas as pd

# --------- Config ---------
st.set_page_config(page_title="GluciPred", page_icon="🍝", layout="wide")

# Essaie d'abord /predict ; tu peux changer dans la sidebar
DEFAULT_API = "https://data-jed-api-glucy-pred.hf.space/predict"

# State
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
    st.caption("Astuce : si /predict ne marche pas, essaie /predict/image")

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
    Essaie 3 formats d'entrée :
      1) multipart/form-data -> files={'file':(...)} + data
      2) JSON -> {'image_base64': '...'}
      3) JSON -> {'image': '...'}
    """
    tries = [
        ("multipart:file", dict(files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                                data={"total_weight_g": total_weight_g})),
        ("json:image_base64", dict(json={"image_base64": _b64(img_bytes),
                                         "total_weight_g": total_weight_g})),
        ("json:image", dict(json={"image": _b64(img_bytes),
                                  "total_weight_g": total_weight_g})),
    ]
    last = None
    for label, kwargs in tries:
        try:
            r = requests.post(api_url, timeout=60, **kwargs)
            if debug:
                st.write(f"⚙️ Try {label} → status {r.status_code}")
                st.write((r.text or "")[:800])
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
    raise last

def parse_response(js: dict):
    """
    Image (b64) : 'segmentated_image' | 'annotated_image' | 'image_b64' | 'image'
    Items : 'labels_and_weights' | 'items' | 'objects' | 'aliments'
            (éventuellement sous 'result' ou 'data')
    Normalise → colonnes: label, percent_total, weight_g, carbs_g, confidence (si dispo)
    """
    if debug:
        st.write("🔎 JSON keys:", list(js.keys()))

    # image annotée
    b64 = js.get("segmentated_image") or js.get("annotated_image") \
         or js.get("image_b64") or js.get("image")
    annotated_bytes = base64.b64decode(b64) if isinstance(b64, str) else None

    # items
    rows = js.get("labels_and_weights") or js.get("items") or js.get("objects") or js.get("aliments")
    if rows is None:
        for k in ("result", "data"):
            d = js.get(k)
            if isinstance(d, dict):
                rows = d.get("labels_and_weights") or d.get("items") or d.get("objects") or d.get("aliments")
                if rows is not None:
                    break

    # liste simple de strings ?
    if isinstance(rows, list) and rows and isinstance(rows[0], str):
        df = pd.DataFrame({"label": rows})
    else:
        df = pd.DataFrame(rows or [])

    # mapping colonnes usuelles → noms standard
    rename_map = {
        "poids": "weight_g",
        "weight": "weight_g",
        "pourcentage": "percent_total",
        "percent": "percent_total",
        "percentage": "percent_total",
        "class": "label",
        "name": "label",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    order = [c for c in ["label", "percent_total", "weight_g", "carbs_g", "confidence"] if c in df.columns]
    if order:
        df = df[order]
    return annotated_bytes, df

# --------- UI ---------
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Importer une image")
    file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
    analyze = st.button("✅ Analyser") if file else None

    if file and analyze:
        img_bytes = file.read()
        with st.spinner("Appel API..."):
            try:
                js = call_api(api_url, img_bytes, total_weight)
                ann, df = parse_response(js)
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
