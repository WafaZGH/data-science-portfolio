import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Getaround — Buffer Simulator", layout="wide")

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    # Ensure numeric
    for c in ["delay_at_checkout_in_minutes", "time_delta_with_previous_rental_in_minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["delay_at_checkout_in_minutes", "time_delta_with_previous_rental_in_minutes"])
    # Helper flags
    df["checkin_type"] = df["checkin_type"].str.lower()
    df["late_return"] = df["delay_at_checkout_in_minutes"] > 0
    df["conflict"] = df["delay_at_checkout_in_minutes"] > df["time_delta_with_previous_rental_in_minutes"]
    return df

df = load_data("delays_clean.csv")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
buffer_min = st.sidebar.slider("Minimum buffer (minutes)", 0, 240, 60, 15)
scope = st.sidebar.selectbox("Scope", ["All cars", "Connect only", "Mobile only"])

if scope == "Connect only":
    view = df[df["checkin_type"] == "connect"].copy()
elif scope == "Mobile only":
    view = df[df["checkin_type"] == "mobile"].copy()
else:
    view = df.copy()

# Safety guard
if view.empty:
    st.warning("No data for the selected scope. Please pick another option.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
st.title("Getaround — Delay Buffer Simulator")
st.caption("Explore trade-offs between conflicts and availability with different buffer policies.")

late_pct     = view["late_return"].mean() * 100
conflict_pct = view["conflict"].mean() * 100
blocked_pct  = (view["time_delta_with_previous_rental_in_minutes"] < buffer_min).mean() * 100
solved_pct   = ((view["conflict"]) & (view["time_delta_with_previous_rental_in_minutes"] < buffer_min)).mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Late Returns (%)", f"{late_pct:.1f}%")
c2.metric("Conflict Rate (%)", f"{conflict_pct:.1f}%")
c3.metric("Rentals Blocked (%)", f"{blocked_pct:.1f}%")
c4.metric("Conflicts Solved (%)", f"{solved_pct:.1f}%")

# -----------------------------
# Q2 — Conflict vs No Conflict
# -----------------------------
no_conflict_pct = (~view["conflict"]).mean() * 100
conflict_pct_now = (view["conflict"]).mean() * 100

q2_df = pd.DataFrame({
    "Status": ["No Conflict", "Conflict"],
    "Percentage": [no_conflict_pct, conflict_pct_now]
})

col_fig, col_note = st.columns([2, 1])
with col_fig:
    fig_q2 = px.bar(q2_df, x="Status", y="Percentage", text="Percentage",
                    title="Q2 • Impact on Next Driver (Conflict vs No Conflict)")
    fig_q2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_q2.update_layout(yaxis_title="%", xaxis_title=None, bargap=0.6,
                         margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_q2, use_container_width=True)

with col_note:
    st.markdown(
        f"""
**How to read**  
- **Conflict**: checkout delay > gap to next rental.  
- Scope = **{scope}** (buffer slider does not affect this chart).  
- Conflict: **{conflict_pct_now:.1f}%** • No conflict: **{no_conflict_pct:.1f}%**.
"""
    )

# -----------------------------
# Q3 — Rentals blocked by buffer threshold
# -----------------------------
thresholds = [30, 60, 120, 180, 240]
blocked_vals = [
    (view["time_delta_with_previous_rental_in_minutes"] < t).mean() * 100
    for t in thresholds
]
q3_df = pd.DataFrame({"Buffer (min)": thresholds, "Blocked (%)": blocked_vals})

col_fig, col_note = st.columns([2, 1])
with col_fig:
    fig_q3 = px.bar(q3_df, x="Buffer (min)", y="Blocked (%)", text="Blocked (%)",
                    title="Q3 • Rentals Blocked by Buffer Threshold")
    fig_q3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_q3.update_layout(yaxis_title="%", bargap=0.6, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_q3, use_container_width=True)

with col_note:
    st.markdown(
        f"""
**What “blocked” means**  
- Rental hidden if gap < buffer.  
- Higher buffer ⇒ **more rentals blocked** (availability ↓).  
- Scope = **{scope}**.
"""
    )

# -----------------------------
# Q4 — Conflicts solved vs buffer (Global denominator)
# -----------------------------
def conflicts_solved_pct_global(df_all: pd.DataFrame, t: int, checkin: str | None = None) -> float:
    mask = (df_all["conflict"]) & (df_all["time_delta_with_previous_rental_in_minutes"] < t)
    if checkin is not None:
        mask = mask & (df_all["checkin_type"] == checkin)
    return mask.mean() * 100  # denominator = all rentals

all_vals_global     = [conflicts_solved_pct_global(df, t) for t in thresholds]
connect_vals_global = [conflicts_solved_pct_global(df, t, checkin="connect") for t in thresholds]

q4a_df = pd.DataFrame({
    "Buffer (min)": thresholds,
    "All Cars": all_vals_global,
    "Connect Only": connect_vals_global
})

col_fig, col_note = st.columns([2, 1])
with col_fig:
    fig_q4a = px.bar(q4a_df, x="Buffer (min)", y=["All Cars", "Connect Only"],
                     barmode="group", title="Q4 • Conflicts Solved vs Buffer (Global %)")
    fig_q4a.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig_q4a.update_layout(yaxis_title="%", bargap=0.45, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_q4a, use_container_width=True)

with col_note:
    st.markdown("""
**Global % denominator** (all rentals).  
- “All Cars” = % of all conflicts solved platform-wide.  
- “Connect Only” = conflicts solved **only among Connect cars**, still divided by **all rentals**.  
- This matches Colab (~8% for Connect).
""")

# -----------------------------
# Q4b — Stacked: Mobile vs Connect contribution (Global denominator)
# -----------------------------
mobile_vals_global  = [conflicts_solved_pct_global(df, t, checkin="mobile") for t in thresholds]
connect_vals_global2 = [conflicts_solved_pct_global(df, t, checkin="connect") for t in thresholds]

q4b_df = pd.DataFrame({
    "Buffer (min)": thresholds,
    "Mobile (solved %, global)": mobile_vals_global,
    "Connect (solved %, global)": connect_vals_global2
})

col_fig, col_note = st.columns([2, 1])
with col_fig:
    fig_q4b = px.bar(
        q4b_df, x="Buffer (min)",
        y=["Mobile (solved %, global)", "Connect (solved %, global)"],
        title="Q4 • Conflicts Solved (Stacked by Check-in Type, Global %)",
        barmode="stack"
    )
    fig_q4b.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig_q4b.update_layout(yaxis_title="%", bargap=0.45, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_q4b, use_container_width=True)

with col_note:
    st.markdown("""
**Stacked contribution (global denominator)**  
- **Height** = total % conflicts solved (same as “All Cars”).  
- Colors split Mobile vs Connect contribution.  
- Since Mobile rentals dominate, most solved conflicts come from Mobile.
""")

# -----------------------------
# About section
# -----------------------------
with st.expander("About this dashboard"):
    st.markdown("""
- **Late Return**: delay at checkout > 0.  
- **Conflict**: delay at checkout > gap to next rental.  
- **Blocked**: rental hidden if gap < buffer.  
- **Conflicts Solved**: conflicts prevented by the buffer policy.  
Use the sidebar to adjust **buffer** and **scope** dynamically.
""")
