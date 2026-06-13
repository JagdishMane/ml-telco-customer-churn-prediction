"""Streamlit client for the telecom churn scoring API.

This is a thin UI. All model logic lives behind the FastAPI service, which this app
calls over HTTP. The form is built dynamically from the API's /schema endpoint, so the
allowed values are never duplicated here.

Run the API first, then:
    streamlit run deployment/client/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Customer Churn Scoring", page_icon=None, layout="centered")
st.title("Telecom Customer Churn Scoring")
st.caption("Group 6, AAI-510. Enter a customer's details to estimate churn risk.")


@st.cache_data(ttl=60)
def get_schema():
    resp = requests.get(f"{API_URL}/schema", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30)
def get_health():
    resp = requests.get(f"{API_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    schema = get_schema()
    health = get_health()
except Exception as exc:
    st.error(
        f"Could not reach the API at {API_URL}. Start it with "
        "`uvicorn deployment.api.main:app --reload` and refresh. "
        f"Details: {exc}"
    )
    st.stop()

FIELDS = schema["fields"]
DEFAULT_THRESHOLD = schema["default_threshold"]
LLM_ON = health.get("llm_configured", False)


def render_result(result: dict):
    """Show a prediction in a consistent way."""
    prob = result["probability"]
    tier = result["risk_tier"]
    cols = st.columns(3)
    cols[0].metric("Churn probability", f"{prob:.1%}")
    cols[1].metric("Risk tier", tier)
    cols[2].metric("Predicted", "Will churn" if result["churn"] else "Will stay")
    st.progress(min(max(prob, 0.0), 1.0))
    if result["churn"]:
        st.warning(
            f"At a threshold of {result['threshold']:.2f}, this customer is flagged as a "
            "likely churner. Consider a retention offer or a contract upgrade."
        )
    else:
        st.success(
            f"At a threshold of {result['threshold']:.2f}, this customer is not flagged as "
            "a likely churner."
        )


def field_input(field: dict, value=None):
    """Render one input widget from a schema field, pre-filled with value if given."""
    name = field["name"]
    if field["type"] == "category":
        choices = field["choices"]
        index = choices.index(value) if value in choices else 0
        return st.selectbox(name, choices, index=index, help=field["description"])

    # Clamp any pre-filled number into the allowed range. A prefill can come from the
    # LLM, which may return an out-of-range value; an above-max value would otherwise
    # crash st.number_input.
    lo, hi = field["min"], field["max"]
    if field["type"] == "int":
        default = int(value) if value is not None else int(lo)
        default = max(int(lo), min(int(hi), default))
        return st.number_input(name, min_value=int(lo), max_value=int(hi),
                               value=default, step=1, help=field["description"])
    try:
        default = float(value) if value is not None else float(lo)
    except (TypeError, ValueError):
        default = float(lo)
    default = max(float(lo), min(float(hi), default))
    return st.number_input(name, min_value=float(lo), max_value=float(hi),
                           value=default, help=field["description"])


tab_single, tab_batch = st.tabs(["Single customer", "Batch CSV"])

# ----------------------------------------------------------------------------------
# Tab 1: single customer
# ----------------------------------------------------------------------------------
with tab_single:
    if "prefill" not in st.session_state:
        st.session_state.prefill = {}

    if LLM_ON:
        with st.expander("Describe the customer in plain text (optional, LLM-guided)",
                         expanded=False):
            st.caption(
                "Type what you know in plain language. The assistant will ask for anything "
                "missing and fill the form below for you to confirm. Tip: send a single "
                "character such as - = * or ? to generate a random sample customer."
            )
            if "chat" not in st.session_state:
                st.session_state.chat = []
                st.session_state.chat_fields = {}

            for msg in st.session_state.chat:
                st.chat_message(msg["role"]).write(msg["content"])

            user_text = st.chat_input(
                "Describe the customer, or send - = * ? for a random one")
            if user_text:
                st.session_state.chat.append({"role": "user", "content": user_text})
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"messages": st.session_state.chat,
                              "fields": st.session_state.chat_fields},
                        timeout=90,
                    )
                    if resp.status_code >= 400:
                        detail = resp.json().get("detail", resp.text)
                        st.session_state.chat.append({
                            "role": "assistant",
                            "content": f"Sorry, the assistant is unavailable right now "
                                       f"({resp.status_code}): {detail}. Please try again, or "
                                       f"fill the form below manually.",
                        })
                    else:
                        data = resp.json()
                        st.session_state.chat.append({"role": "assistant", "content": data["reply"]})
                        st.session_state.chat_fields = data["fields"]
                        st.session_state.prefill = data["fields"]
                        if data["complete"]:
                            st.success("All fields collected. Review the form below and score.")
                except Exception as exc:
                    st.error(f"Chat failed: {exc}")
                st.rerun()

    st.subheader("Customer details")
    prefill = st.session_state.prefill
    with st.form("single_customer"):
        values = {}
        # Lay the fields out in two columns to use less vertical space.
        columns = st.columns(2)
        for i, field in enumerate(FIELDS):
            with columns[i % 2]:
                values[field["name"]] = field_input(field, prefill.get(field["name"]))
        threshold = st.slider("Decision threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01,
                              help="Lower catches more churners at the cost of more false alarms.")
        submitted = st.form_submit_button("Score customer")

    if submitted:
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"customer": values, "threshold": threshold},
                timeout=30,
            )
            if resp.status_code == 422:
                st.error(f"Invalid input: {resp.json().get('detail')}")
            else:
                resp.raise_for_status()
                render_result(resp.json())
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")

# ----------------------------------------------------------------------------------
# Tab 2: batch CSV
# ----------------------------------------------------------------------------------
with tab_batch:
    required = ", ".join(f["name"] for f in FIELDS)
    st.write("Upload a CSV with one customer per row and these columns:")
    st.code(required, language="text")

    threshold_b = st.slider("Decision threshold ", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01,
                            key="batch_threshold")
    upload = st.file_uploader("Customer CSV", type=["csv"])

    if upload is not None and st.button("Score file"):
        try:
            resp = requests.post(
                f"{API_URL}/predict/batch",
                files={"file": (upload.name, upload.getvalue(), "text/csv")},
                params={"threshold": threshold_b},
                timeout=120,
            )
            if resp.status_code >= 400:
                st.error(resp.json().get("detail", "Scoring failed."))
            else:
                data = resp.json()
                results = pd.DataFrame(data["results"])
                st.success(f"Scored {data['count']} customers at threshold {data['threshold']:.2f}.")
                st.metric("Flagged as likely churners", int(results["churn"].sum()))
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download results CSV",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
