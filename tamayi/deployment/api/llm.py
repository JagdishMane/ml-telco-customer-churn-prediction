"""Optional LLM-driven conversational collection of the seven customer fields.

This is provider-agnostic: it talks to any OpenAI-compatible chat endpoint, so it
works with DeepSeek or Qwen by setting LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL. The
whole module is optional. If no key is configured the API never calls it.

The model is asked to reply with a small JSON object on every turn: a natural-language
message to the user, the fields it has gathered so far, and whether all seven are
complete. We parse that JSON server-side; the values are still validated by Pydantic
before any scoring happens, so a malformed extraction cannot reach the model.
"""

from __future__ import annotations

import json
import os

from .model import RAW_FIELDS

_FIELD_BY_NAME = {f["name"]: f for f in RAW_FIELDS}


def _valid_value(name, value):
    """Return the coerced value if it is valid for the field, else None.

    Drops anything the LLM should not have sent: unknown field names, null values,
    categories outside the allowed set, and numbers outside the documented range.
    """
    field = _FIELD_BY_NAME.get(name)
    if field is None or value is None:
        return None
    if field["type"] == "category":
        return value if value in field["choices"] else None
    try:
        number = int(value) if field["type"] == "int" else float(value)
    except (TypeError, ValueError):
        return None
    if number < field["min"] or number > field["max"]:
        return None
    return number


_SCHEMA_LINES = []
for f in RAW_FIELDS:
    if f["type"] == "category":
        allowed = ", ".join(f["choices"])
        _SCHEMA_LINES.append(f"- {f['name']} ({f['description']}). One of: {allowed}.")
    else:
        _SCHEMA_LINES.append(
            f"- {f['name']} ({f['description']}). A {f['type']} "
            f"between {f['min']} and {f['max']}."
        )
_FIELD_SCHEMA = "\n".join(_SCHEMA_LINES)

SYSTEM_PROMPT = f"""You help a telecom analyst gather the details needed to score a \
customer for churn. There are seven fields:

{_FIELD_SCHEMA}

Rules:
- Read whatever the user has typed and extract any fields you can.
- Ask a brief, friendly follow-up question for the fields that are still missing or \
invalid. Ask for several at once when natural; do not interrogate one at a time.
- Map loose phrasing to the allowed category values (e.g. "pays by credit card" maps to \
Credit card). Never invent a value the user did not imply.
- Keep numbers within the stated ranges. If a value is out of range, point it out and \
ask again.
- TotalCharges is optional. If the user does not give it, do NOT ask for it: leave it out \
and it will be set automatically to Tenure times MonthlyCharges.
- Accept any value that is within its range. Do not question whether the values are \
consistent with one another (for example, do not object that TotalCharges looks low \
relative to Tenure and MonthlyCharges).
- In "fields", include only fields you have a valid value for. Omit a field entirely if \
it is unknown or invalid. Never set a field to null.

The customer is ready to score once the other six fields (every field except \
TotalCharges) are present and valid.

You must reply with ONLY a JSON object, no prose outside it, of the form:
{{"message": "<your message to the user>", "fields": {{<only the valid fields so far>}}, \
"complete": <true once the six required fields are present and valid>}}
"""

# A single non-alphanumeric character (e.g. - = * ?) is a shortcut: instead of collecting
# from the user, invent a random plausible customer and fill the whole form.
RANDOMIZE_PROMPT = f"""Invent ONE realistic, randomly varied telecom customer and return \
all seven fields. Make a fresh, distinct profile each time, with believable combinations \
of age, tenure, charges, contract, and payment method. The fields are:

{_FIELD_SCHEMA}

Set TotalCharges roughly to Tenure times MonthlyCharges. Keep every value within its range.

Reply with ONLY a JSON object, no prose outside it, of the form:
{{"message": "<one short sentence summarising the generated customer>", \
"fields": {{<all seven fields>}}, "complete": true}}
"""


def _is_randomize_trigger(text: str) -> bool:
    """A single non-alphanumeric character triggers random customer generation."""
    stripped = (text or "").strip()
    return len(stripped) == 1 and not stripped.isalnum()


def _client():
    """Build an OpenAI-compatible client from environment configuration."""
    from openai import OpenAI

    return OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        # Bound the wait and retry a couple of times, so a transient upstream hiccup
        # (e.g. a 502 from the provider) does not immediately fail the request.
        timeout=float(os.getenv("LLM_TIMEOUT", "45")),
        max_retries=2,
    )


def collect(messages: list[dict], fields: dict) -> dict:
    """Run one conversational turn.

    messages: prior turns as [{"role": "user"|"assistant", "content": str}].
    fields: fields gathered so far.
    Returns {"reply": str, "fields": dict, "complete": bool}.
    """
    # Shortcut: a lone non-alphanumeric character means "make one up for me".
    last_user = next((m["content"] for m in reversed(messages)
                      if m.get("role") == "user"), "")
    if _is_randomize_trigger(last_user):
        return _random_customer()

    model_name = os.getenv("LLM_MODEL", "deepseek-chat")

    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    if fields:
        chat.append({
            "role": "system",
            "content": f"Fields gathered so far: {json.dumps(fields)}",
        })
    chat.extend(messages)

    completion = _client().chat.completions.create(
        model=model_name,
        messages=chat,
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"reply": raw, "fields": fields, "complete": False}

    return _finalize(parsed, fields)


def _random_customer() -> dict:
    """Ask the model to invent a random, fully populated customer."""
    model_name = os.getenv("LLM_MODEL", "deepseek-chat")
    completion = _client().chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": RANDOMIZE_PROMPT}],
        # Higher temperature so repeated triggers produce different customers.
        temperature=1.2,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"reply": raw, "fields": {}, "complete": False}
    return _finalize(parsed, {})


def _finalize(parsed: dict, fields: dict) -> dict:
    """Merge, validate, derive TotalCharges, and decide completeness."""
    # Merge, then keep only valid values. The model sometimes returns an out-of-range or
    # malformed value (e.g. a monthly charge of 800) even while flagging it to the user.
    # Dropping it here means the form is never pre-filled with a bad value: that field
    # simply stays unanswered until the user supplies a valid one.
    merged = {**fields, **(parsed.get("fields") or {})}
    merged = {k: clean for k, v in merged.items()
              if (clean := _valid_value(k, v)) is not None}

    # TotalCharges is optional. When it is missing but tenure and monthly charges are
    # known, default it to Tenure times MonthlyCharges rather than nagging the user.
    if ("TotalCharges" not in merged
            and "Tenure" in merged and "MonthlyCharges" in merged):
        derived = round(float(merged["Tenure"]) * float(merged["MonthlyCharges"]), 2)
        if _valid_value("TotalCharges", derived) is not None:
            merged["TotalCharges"] = derived

    # Complete once every field is present. Trust the model's flag, but also confirm
    # server-side so a derived TotalCharges counts toward completion.
    complete = bool(parsed.get("complete", False)) or all(
        name in merged for name in _FIELD_BY_NAME
    )
    return {
        "reply": parsed.get("message", ""),
        "fields": merged,
        "complete": complete,
    }
