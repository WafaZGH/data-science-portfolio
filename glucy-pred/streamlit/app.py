import streamlit as st
from PIL import Image
from io import BytesIO
import base64, requests, pandas as pd

# -------- Settings --------
st.set_page_config(page_title="GluciPred", page_icon="🍝", layout="wide")
DEFAULT_API = "https://data-jed-api-glucipred.hf.space/predict/image"  # <-- your API

# Session state
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "annotated_bytes" not in st.session_state:
    st.session_state.annotated_bytes = None

# -------- Sidebar --------
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API endpoint", DEFAULT_API)
    total_weight = st.slider("Estimated plate weight (g)", 150, 800, 350, 10)
    st.caption("This total weight is used by the backend to distribute grams per item.")

    st.write("---")
    st.caption("Tip: Make sure your API accepts either multipart upload or base64 JSON.")

# -------- Header --------
st.markdown(
    """
    <div style="text-align:center;margin-top:10px;margin-bottom:20px">
      <h1 style="font-size:3rem">🍝 <span style="color:#4CAF50">GLUCY-Pred</span></h1>
      <p>Upload a plate photo → segmentation → weights & carbs</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Optional hero GIF (put the file next to this app or remove this line)
try:
    st.image("glucy_final_loop.gif", use_container_width=True)
except Exception:
    pass

# -------- Helpers --------
def _b64(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def call_api(api_url: str, img_bytes: bytes, total_weight_g: int):
    """
    Try two common payloads:
      1) multipart/form-data (files=...)
      2) JSON with base64 (json={"image": "...", "total_weight_g": ...})
    """
    # 1) multipart
    try:
        files = {"file": ("upload.jpg", img_bytes, "image/jpeg")}
        data = {"total_weight_g": total_weight_g}
        r = requests.post(api_url, files=files, data=data, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        # 2) JSON base64
        payload = {"image": _b64(img_bytes), "total_weight_g": total_weight_g}
        r = requests.post(api_url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

def parse_response(js: dict):
    """
    Accept a few variants from the backend:
      - annotated image key: 'segmentated_image' | 'annotated_image' | 'image'
      - items list key: 'labels_and_weights' | 'items' | 'objects' | within 'result'/'data'
    Normalize columns to: label, percent_total, weight_g, carbs_g, confidence (when present).
    """
    # image
    img_b64 = js.get("segmentated_image") or js.get("annotated_image") or js.get("image")
    annotated_bytes = base64.b64decode(img_b64) if isinstance(img_b64, str) else None

    # items
    rows = js.get("labels_and_weights") or js.get("items") or js.get("objects")
    if rows is None and isinstance(js, dict):
        for k in ("result", "data"):
            if isinstance(js.get(k), dict):
                rows = js[k].get("labels_and_weights") or js[k].get("items") or js[k].get("objects")
                if rows is not None:
                    break
    df = pd.DataFrame(rows or [])

    # normalize names
    rename_map = {
        "poids": "weight_g",
        "weight": "weight_g",
        "percent": "percent_total",
        "percentage": "percent_total",
        "class": "label",
        "name": "label",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # pretty order
    cols = [c for c in ["label", "percent_total", "weight_g", "carbs_g", "confidence"] if c in df.columns]
    if cols:
        df = df[cols]

    return annotated_bytes, df

# -------- UI --------
left, right = st.columns([1, 1])

with left:
    st.subheader("📤 Upload a photo")
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    analyze = st.button("✅ Analyze") if file else None

    if file and analyze:
        img_bytes = file.read()
        with st.spinner("Calling API…"):
            try:
                js = call_api(api_url, img_bytes, total_weight)
                annotated, df = parse_response(js)
                st.session_state.annotated_bytes = annotated
                st.session_state.results_df = df
                st.success("Done!")
            except Exception as e:
                st.error(f"API error: {e}")

    if file:
        st.caption("Original")
        st.image(Image.open(file), use_container_width=True)

with right:
    st.subheader("🧾 Results")
    df = st.session_state.results_df
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            if "weight_g" in df.columns:
                st.metric("Total estimated weight (g)", f"{df['weight_g'].sum():.0f}")
        with c2:
            if "carbs_g" in df.columns:
                st.metric("Total carbs (g)", f"{df['carbs_g'].sum():.1f}")
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode(), "results.csv", "text/csv")
    else:
        st.info("Upload an image and click Analyze to see results here.")

    if st.session_state.annotated_bytes:
        st.caption("Segmentation")
        st.image(Image.open(BytesIO(st.session_state.annotated_bytes)), use_container_width=True)
