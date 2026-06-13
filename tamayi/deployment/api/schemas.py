"""Pydantic request and response models for the churn API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """One customer's raw details, validated against the documented ranges."""

    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Gender: Literal["Female", "Male", "Other"]
    Tenure: int = Field(..., ge=1, le=72,
                        description="Months with the company (>= 1 to avoid divide-by-zero)")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly bill in USD")
    TotalCharges: float = Field(..., ge=0, description="Total billed over tenure in USD")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal["Bank transfer", "Credit card", "Electronic check", "Mailed check"]


class PredictRequest(BaseModel):
    """A single-customer scoring request with an optional decision threshold."""

    customer: CustomerInput
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0,
                                       description="Override the default churn threshold")


class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    risk_tier: str
    threshold: float


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    fields: dict = Field(default_factory=dict,
                         description="Fields collected so far")


class ChatResponse(BaseModel):
    reply: str
    fields: dict
    complete: bool
