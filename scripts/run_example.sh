#!/usr/bin/env bash
set -euo pipefail

oracle-study validate --input data/examples/interactions.csv
oracle-study analyze \
  --input data/examples/interactions.csv \
  --output-dir results/example \
  --lambda-cost 0.10
