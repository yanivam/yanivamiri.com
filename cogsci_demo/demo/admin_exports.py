"""CSV export helpers for admin downloads."""

from __future__ import annotations

import csv
import io

from demo.storage import export_all_emails, export_all_sessions


def build_contacts_csv() -> str:
    rows = export_all_emails()
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["session_id", "name", "email", "consent", "created_at"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def build_choices_csv() -> str:
    rows = export_all_sessions()
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "session_id",
            "created_at",
            "drug_1",
            "drug_2",
            "factor_1",
            "factor_2",
            "factor_3",
            "background",
        ],
    )
    writer.writeheader()
    for row in rows:
        drugs = row["selected_drugs"]
        factors = row["selected_factors"]
        writer.writerow(
            {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "drug_1": drugs[0] if len(drugs) > 0 else "",
                "drug_2": drugs[1] if len(drugs) > 1 else "",
                "factor_1": factors[0] if len(factors) > 0 else "",
                "factor_2": factors[1] if len(factors) > 1 else "",
                "factor_3": factors[2] if len(factors) > 2 else "",
                "background": row.get("background") or "",
            }
        )
    return buffer.getvalue()


def admin_summary() -> dict:
    sessions = export_all_sessions()
    contacts = export_all_emails()
    latest = sessions[-1]["created_at"] if sessions else None
    return {
        "sessions": len(sessions),
        "contacts": len(contacts),
        "latest_session_at": latest,
    }
