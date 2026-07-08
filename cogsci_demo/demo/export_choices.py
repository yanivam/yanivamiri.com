#!/usr/bin/env python3
"""Export anonymous booth choices to CSV for post-conference analysis."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.storage import export_all_sessions


def main() -> None:
    out_path = Path(__file__).resolve().parent / "data" / "booth_choices_export.csv"
    rows = export_all_sessions()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
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
    print(f"Wrote {len(rows)} sessions to {out_path}")


if __name__ == "__main__":
    main()
