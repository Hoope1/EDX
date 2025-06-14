#!/usr/bin/env bash
set -euo pipefail
black --check edge_detection_tool scripts tests
isort --check-only edge_detection_tool scripts tests
flake8 edge_detection_tool scripts tests
mypy edge_detection_tool
pytest -q
