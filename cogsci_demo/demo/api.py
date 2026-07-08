"""FastAPI server for booth QR demo."""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from demo.admin_exports import admin_summary, build_choices_csv, build_contacts_csv
from demo.constants import BACKGROUND_OPTIONS, DRUG_DISPLAY
from demo.simulation import get_scenario_preview, run_comparison
from demo.storage import get_session, init_db, save_email, save_email_for_session, save_session

STATIC_DIR = Path(__file__).resolve().parent / "static"
ADMIN_TOKEN = os.getenv("DEMO_ADMIN_TOKEN")
_admin_bearer = HTTPBearer(auto_error=False)


def _require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_admin_bearer),
) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Admin exports are disabled. Set DEMO_ADMIN_TOKEN on the server.",
        )
    if credentials is None or credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Attention Policy Booth Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    selected_drugs: list[str] = Field(..., min_length=2, max_length=2)
    selected_factors: list[str] = Field(..., min_length=3, max_length=3)
    name: str | None = None
    email: str | None = None
    consent: bool = False
    background: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Name is too short")
        return cleaned

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()):
            raise ValueError("Invalid email address")
        return value.strip().lower()


class ConnectRequest(BaseModel):
    session_id: str
    name: str
    email: str
    consent: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Name is required")
        return cleaned

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()):
            raise ValueError("Invalid email address")
        return value.strip().lower()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    os.environ.setdefault("OVERRIDE_RNG_SEED", "42")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenario")
def scenario() -> dict:
    return get_scenario_preview()


@app.get("/api/backgrounds")
def backgrounds() -> dict:
    labels = {
        "cognitive_science": "Cognitive science",
        "ai_ml": "AI / ML",
        "healthcare": "Healthcare",
        "operations": "Operations / supply chain",
        "other": "Other",
    }
    return {
        "options": [{"id": key, "label": labels[key]} for key in BACKGROUND_OPTIONS]
    }


@app.post("/api/run")
def run_demo(request: RunRequest) -> dict:
    if request.background and request.background not in BACKGROUND_OPTIONS:
        raise HTTPException(status_code=400, detail="Invalid background option")
    if request.email and not request.consent:
        raise HTTPException(status_code=400, detail="Consent required to save email")
    if request.email and not request.name:
        raise HTTPException(status_code=400, detail="Name required with email")

    try:
        result = run_comparison(request.selected_drugs, request.selected_factors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_session(
        session_id=result["session_id"],
        selected_drugs=request.selected_drugs,
        selected_factors=request.selected_factors,
        background=request.background,
        result=result,
    )

    if request.email and request.consent:
        save_email(result["session_id"], request.email, request.consent, name=request.name)

    return result


@app.get("/api/results/{session_id}")
def get_results(session_id: str) -> dict:
    record = get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return record["result"]


@app.post("/api/connect")
def connect(request: ConnectRequest) -> dict[str, str]:
    if not request.consent:
        raise HTTPException(status_code=400, detail="Consent required")
    try:
        save_email_for_session(
            request.session_id,
            request.email,
            request.consent,
            name=request.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/api/admin/summary")
def admin_stats(_: None = Depends(_require_admin)) -> dict:
    return admin_summary()


@app.get("/api/admin/contacts.csv")
def admin_contacts_csv(_: None = Depends(_require_admin)) -> Response:
    return Response(
        content=build_contacts_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="booth_contacts.csv"'},
    )


@app.get("/api/admin/choices.csv")
def admin_choices_csv(_: None = Depends(_require_admin)) -> Response:
    return Response(
        content=build_choices_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="booth_choices.csv"'},
    )


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/r/{session_id}")
def share_result(session_id: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
