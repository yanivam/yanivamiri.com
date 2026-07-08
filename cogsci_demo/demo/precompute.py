#!/usr/bin/env python3
"""Train LearnedAttentionAgent on booth scenario (run once before conference)."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from demo.simulation import train_and_save_learner


def main() -> None:
    print("Training LearnedAttentionAgent on booth scenario (REINFORCE)...")
    payload = train_and_save_learner(training_weeks=25)
    print(f"Saved weights to {payload['weights_path']}")
    print("Learned weights:", payload["attention_weights"])


if __name__ == "__main__":
    main()
