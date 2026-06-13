"""FastAPI service for telecom customer churn scoring.

Endpoints:
- GET  /health        liveness and whether the optional LLM is configured
- GET  /schema        the seven raw input fields with types and allowed values
- POST /predict       score a single customer
- POST /predict/batch score a CSV upload of many customers
- POST /chat          optional LLM-guided field collection (503 when no LLM configured)
"""

from __future__ import annotations

import io
import logging

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from . import model as model_core
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
)

app = FastAPI(
    title="Telecom Churn Scoring API",
    description="Scores telecom customers for churn using the trained Group 6 model.",
    version="1.0.0",
)

# The Streamlit client runs as a separate process, so allow cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm_configured=model_core.llm_configured())


@app.get("/schema")
def schema():
    """Field metadata that drives the client form and the LLM prompt."""
    return {
        "fields": model_core.RAW_FIELDS,
        "default_threshold": model_core.DEFAULT_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    threshold = request.threshold if request.threshold is not None else model_core.DEFAULT_THRESHOLD
    try:
        result = model_core.predict([request.customer.model_dump()], threshold)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PredictionResponse(**result)


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), threshold: float | None = None):
    """Score a CSV containing the seven raw columns, one customer per row."""
    threshold = threshold if threshold is not None else model_core.DEFAULT_THRESHOLD

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    missing = [c for c in model_core.RAW_FIELD_NAMES if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {', '.join(missing)}. "
                   f"Expected: {', '.join(model_core.RAW_FIELD_NAMES)}.",
        )

    records = df[model_core.RAW_FIELD_NAMES].to_dict(orient="records")
    try:
        predictions = model_core.predict(records, threshold)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}")

    rows = []
    for record, prediction in zip(records, predictions):
        rows.append({**record, **prediction})
    return {"count": len(rows), "threshold": threshold, "results": rows}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Optional LLM-guided collection of the seven fields from free text."""
    if not model_core.llm_configured():
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured. Set LLM_API_KEY to enable the chat path.",
        )

    from . import llm

    messages = [m.model_dump() for m in request.messages]
    try:
        outcome = llm.collect(messages, request.fields)
    except Exception as exc:
        # Log the full cause server-side so a transient upstream failure is diagnosable,
        # and pass a readable message back to the client.
        logging.getLogger("uvicorn.error").exception("LLM /chat call failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    return ChatResponse(
        reply=outcome["reply"],
        fields=outcome["fields"],
        complete=outcome["complete"],
    )
