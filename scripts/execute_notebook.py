"""Ejecuta el notebook completo desde un kernel limpio y guarda sus salidas."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Lab1_Entrenamiento_MLP_California_Housing.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="lab1dl")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK)
    print(f"Notebook ejecutado sin errores: {NOTEBOOK}")


if __name__ == "__main__":
    main()

