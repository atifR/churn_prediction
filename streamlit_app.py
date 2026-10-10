import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# the FastAPI app must be running; override with the CHURN_API_URL env var if it is not on the default port
API_URL = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")

YES_NO = ["Yes", "No"]
YES_NO_NO_INTERNET = ["Yes", "No", "No internet service"]

# same allowed values as the Customer model in FastAPI_main.py
CHOICES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": YES_NO_NO_INTERNET,
    "OnlineBackup": YES_NO_NO_INTERNET,
    "DeviceProtection": YES_NO_NO_INTERNET,
    "TechSupport": YES_NO_NO_INTERNET,
    "StreamingTV": YES_NO_NO_INTERNET,
    "StreamingMovies": YES_NO_NO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# form layout: one list of fields per column
COLUMNS = [
    ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"],
    ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup"],
    ["DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract"],
    ["PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
]


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def load_samples():
    st.session_state.samples = api_get("/sample-customers", n=5)
    st.session_state.sample_idx = 0
    fill_form()


def fill_form():
    # copy the selected sample into the widget values
    sample = st.session_state.samples[st.session_state.sample_idx]
    for field, value in sample.items():
        st.session_state[field] = value
    st.session_state.pop("prediction", None)


@st.cache_data(ttl=600)
def load_feature_importance() -> pd.DataFrame:
    df = pd.DataFrame(api_get("/feature-importance", top=50))
    # drop the "num__" / "cat__" prefix added by the ColumnTransformer
    df["feature"] = df["feature"].str.split("__").str[-1]
    return df


st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Churn Prediction")

try:
    if "samples" not in st.session_state:
        load_samples()
    importance = load_feature_importance()
except requests.RequestException as e:
    st.error(
        f"Could not reach the API at {API_URL}. Start it with "
        f"`uv run uvicorn FastAPI_main:app` and reload this page.\n\n{e}"
    )
    st.stop()

st.subheader("Customer")
pick, refresh = st.columns([5, 1], vertical_alignment="bottom")
labels = [
    f"Sample {i + 1}: {s['gender']}, {s['tenure']} months, {s['Contract']}"
    for i, s in enumerate(st.session_state.samples)
]
pick.radio(
    "Start from a sample customer",
    range(len(labels)),
    format_func=labels.__getitem__,
    key="sample_idx",
    on_change=fill_form,
)
refresh.button("New samples", on_click=load_samples, width="stretch")

with st.form("customer"):
    for col, fields in zip(st.columns(len(COLUMNS)), COLUMNS):
        for field in fields:
            if field == "SeniorCitizen":
                col.selectbox(field, CHOICES[field], key=field, format_func=lambda v: "Yes" if v else "No")
            elif field in CHOICES:
                col.selectbox(field, CHOICES[field], key=field)
            elif field == "tenure":
                col.number_input("tenure (months)", min_value=0, step=1, key=field)
            else:
                col.number_input(field, min_value=0.0, step=1.0, format="%.2f", key=field)
    submitted = st.form_submit_button("OK", type="primary")

if submitted:
    customer = {field: st.session_state[field] for fields in COLUMNS for field in fields}
    try:
        r = requests.post(f"{API_URL}/predict", json=customer, timeout=10)
        r.raise_for_status()
        st.session_state.prediction = r.json()
    except requests.RequestException as e:
        st.session_state.pop("prediction", None)
        st.error(f"Prediction failed: {e}")
if "prediction" in st.session_state:
    p = st.session_state.prediction
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.metric(
                label="Prediction",
                value="Will churn" if p["churn"] else "Will stay",
                delta="High Risk" if p["churn"] else "Low Risk",
                delta_color="inverse",  # Red if churning, green if staying
            )

    with col2:
        with st.container(border=True):
            st.metric(
                label="Decision score",
                value=f"{p['score']:+.3f}",
            )
    st.caption("The score is the SVC decision value, not a probability: above 0 means churn, and further from 0 means more confident.")

st.subheader("Feature importance")
top = st.slider("Features to show", 3, len(importance), min(10, len(importance)))
data = importance.head(top)

dark = st.context.theme.type == "dark"
bars = alt.Chart(data).encode(
    y=alt.Y("feature:N", sort=None, title=None),
    # leave room on the right for the value labels
    x=alt.X(
        "importance_pct:Q",
        title="Importance (%)",
        axis=alt.Axis(tickCount=6),
        scale=alt.Scale(domain=[0, data["importance_pct"].max() * 1.08], nice=False),
    ),
    tooltip=[
        alt.Tooltip("feature:N", title="Feature"),
        alt.Tooltip("importance_pct:Q", title="Importance (%)", format=".2f"),
    ],
)
chart = (
    bars.mark_bar(color="#3987e5" if dark else "#2a78d6", cornerRadiusEnd=4, height={"band": 0.6})
    + bars.mark_text(align="left", dx=6, color="#c3c2b7" if dark else "#52514e").encode(
        text=alt.Text("importance_pct:Q", format=".1f")
    )
).properties(height=32 * top + 40)
st.altair_chart(chart, width="stretch")

