#!/usr/bin/env python3
"""Export booth contacts (name + email) to CSV."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.storage import export_all_emails


def main() -> None:
    out_path = Path(__file__).resolve().parent / "data" / "booth_contacts_export.csv"
    rows = export_all_emails()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["session_id", "name", "email", "consent", "created_at"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} contacts to {out_path}")


if __name__ == "__main__":
    main()
