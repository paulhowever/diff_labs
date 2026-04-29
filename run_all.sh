#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_NOTEBOOKS=false

if [[ "${1:-}" == "--open-notebooks" ]]; then
  OPEN_NOTEBOOKS=true
fi

echo "==> Lab1: generate data"
dotnet run --project "$ROOT_DIR/lab1/src/Lab1.App/Lab1.App.csproj"

echo "==> Lab2: generate data"
dotnet run --project "$ROOT_DIR/lab2/src/Lab2.App/Lab2.App.csproj"

echo "==> Lab1: tests"
dotnet test "$ROOT_DIR/lab1/tests/Lab1.Tests/Lab1.Tests.csproj"

echo "==> Lab2: tests"
dotnet test "$ROOT_DIR/lab2/tests/Lab2.Tests/Lab2.Tests.csproj"

echo "==> Done: all calculations and tests passed."

if [[ "$OPEN_NOTEBOOKS" == true ]]; then
  echo "==> Opening Jupyter notebooks..."
  jupyter notebook \
    "$ROOT_DIR/lab1/notebooks/analysis.ipynb" \
    "$ROOT_DIR/lab2/notebooks/analysis.ipynb"
else
  echo "Tip: run './run_all.sh --open-notebooks' to open notebooks."
fi
