#!/bin/bash
set -e

echo "Running K=1..."
.venv/bin/python -m src.evaluation.beam_search_runner --k 1

echo "Running K=3..."
.venv/bin/python -m src.evaluation.beam_search_runner --k 3

echo "Running K=5..."
.venv/bin/python -m src.evaluation.beam_search_runner --k 5

echo "Running K=10..."
.venv/bin/python -m src.evaluation.beam_search_runner --k 10

echo "Running K=20..."
.venv/bin/python -m src.evaluation.beam_search_runner --k 20

echo "Running Analysis..."
PYTHONPATH=. .venv/bin/python scripts/run_EXP018A_analysis.py
PYTHONPATH=. .venv/bin/python scripts/analyze_EXP018A_statistics.py

echo "Done!"
