"""Comprueba los invariantes academicos y tecnicos de la entrega."""

from __future__ import annotations

import json
import math
from pathlib import Path

import nbformat
import pandas as pd
import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
NOTEBOOK = ROOT / "Lab1_Entrenamiento_MLP_California_Housing.ipynb"
PDF = ROOT / "output" / "pdf" / "Laboratorio_1_Ian_Cumes_23236.pdf"


def main() -> None:
    required = [NOTEBOOK, PDF, ARTIFACTS / "experiments.csv", ARTIFACTS / "histories.json", ARTIFACTS / "final_summary.json"]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Faltan artefactos: {missing}"

    results = pd.read_csv(ARTIFACTS / "experiments.csv")
    histories = json.loads((ARTIFACTS / "histories.json").read_text(encoding="utf-8"))
    summary = json.loads((ARTIFACTS / "final_summary.json").read_text(encoding="utf-8"))
    assert len(results) == 17 and results["id"].is_unique
    numeric = results[["val_mse", "val_mae", "val_rmse", "training_seconds"]].to_numpy().ravel()
    assert all(math.isfinite(float(value)) for value in numeric)
    for _, row in results.iterrows():
        history = histories[str(int(row["id"]))]
        assert len(history["train_mse"]) == int(row["epochs"])
        assert len(history["train_objective"]) == int(row["epochs"])
        assert len(history["val_mse"]) == int(row["epochs"])
    ranked = results.sort_values(["val_rmse", "val_mae", "parameters", "training_seconds"])
    assert int(ranked.iloc[0]["id"]) == int(summary["best_experiment"]) == 8

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    reader = PdfReader(PDF)
    assert len(reader.pages) == 3
    with pdfplumber.open(PDF) as document:
        assert len(document.pages) == 3
        text = "\n".join(page.extract_text() or "" for page in document.pages)
    for expected in [
        "Ian Cumes",
        "23236",
        "github.com/iancumes/Lab1DeepLearning",
        "nn.Linear",
        "SmoothL1Loss",
        "SGD",
        "Adam",
        "RMSprop",
        "17 corridas",
        "TEST RMSE",
    ]:
        assert expected in text, f"Contenido ausente en PDF: {expected}"
    print("Validacion completa: 17 experimentos, notebook ejecutado y PDF de 3 paginas.")


if __name__ == "__main__":
    main()

