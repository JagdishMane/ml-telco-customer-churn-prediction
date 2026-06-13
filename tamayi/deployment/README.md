# Churn Scoring Deployment

A small two-part deployment that turns the trained Group 6 churn model into a usable
application:

- `api/` is a FastAPI service that loads the trained artifacts and scores customers.
- `client/` is a Streamlit app that calls the API over HTTP. It offers a manual form, an
  optional LLM-guided plain-text path, and batch CSV scoring.

The model is the Logistic Regression saved in `tamayi/models/best_model.pkl`, served
together with `tamayi/models/preprocessor.pkl`. No re-training happens here. The service
engineers the four derived features (AvgChargePerMonth, IsShortTenure, IsHighMonthly,
SeniorFlag) exactly as notebook 03, applies the saved preprocessor, then runs the model.

## The seven input fields

| Field | Type | Allowed values or range |
| --- | --- | --- |
| Age | int | 18 to 100 |
| Gender | category | Female, Male, Other |
| Tenure | int | 1 to 72 (months) |
| MonthlyCharges | float | about 10 to 150 |
| TotalCharges | float | about 0 to 11000 |
| Contract | category | Month-to-month, One year, Two year |
| PaymentMethod | category | Bank transfer, Credit card, Electronic check, Mailed check |

CustomerID and Churn are not inputs: the first is an identifier, the second is the target.

## Running it

Dependencies are managed with [uv](https://docs.astral.sh/uv/) from the `tamayi/` folder.
Run `uv sync` once there, then run the two commands below from that same `tamayi/` folder
(the parent of this `deployment/` folder) in separate terminals.

### 1. API

```bash
uv run uvicorn deployment.api.main:app --reload
```

The API serves on http://localhost:8000. Interactive docs are at http://localhost:8000/docs.
If the model artifacts are not at the default `tamayi/models` location, set `MODEL_DIR`.

### 2. Client

```bash
uv run streamlit run deployment/client/app.py
```

The client opens in the browser and talks to the API at http://localhost:8000 by default
(override with `API_URL`).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health | Liveness, and whether the LLM is configured |
| GET | /schema | Field metadata that drives the form and the LLM |
| POST | /predict | Score a single customer |
| POST | /predict/batch | Score a CSV upload of many customers |
| POST | /chat | Optional LLM-guided field collection (503 if no LLM) |

Example single prediction:

```bash
curl -s http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "customer": {
    "Age": 32, "Gender": "Male", "Tenure": 3, "MonthlyCharges": 120.0,
    "TotalCharges": 360.0, "Contract": "Month-to-month", "PaymentMethod": "Electronic check"
  },
  "threshold": 0.40
}'
```

## Decision threshold

The default threshold is 0.40, the recall-favoring operating point recommended in notebook
05, because missing a likely churner costs more than a false alarm. Both the single and
batch views expose a slider to adjust it.

## Optional LLM

The LLM path is entirely optional. With no API key the manual form and batch upload work
end to end. To enable the conversational path, copy `api/.env.example` to `api/.env` and set
`LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. Any OpenAI-compatible endpoint works, e.g.
DeepSeek or Qwen. When configured, `/health` reports `llm_configured: true` and the client
shows a plain-text chat box that collects the seven fields and pre-fills the form.

