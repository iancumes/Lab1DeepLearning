"""Construye el notebook entregable a partir de celdas versionables."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Lab1_Entrenamiento_MLP_California_Housing.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# Laboratorio 1 - Entrenamiento de redes neuronales

**Curso:** CC3092 - Deep Learning y Sistemas Inteligentes  
**Estudiante:** Ian Cumes  
**Carne:** 23236  
**Repositorio:** https://github.com/iancumes/Lab1DeepLearning

El objetivo es predecir el valor mediano de vivienda de distritos censales de California mediante un MLP de regresion en PyTorch. La seleccion del modelo se realiza exclusivamente con validacion; test permanece aislado hasta el final.
"""),
    md("""
## 1. Reproducibilidad y librerias

El experimento usa CPU y semilla 42. Se fijan las fuentes de aleatoriedad y se solicitan algoritmos deterministas para que las comparaciones entre configuraciones sean justas.
"""),
    code("""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data import SEED, TARGET, load_housing, quality_report, split_and_scale, validate_bundle

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8-whitegrid")

ARTIFACTS = Path("artifacts")
FIGURES = ARTIFACTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
print(f"PyTorch {torch.__version__} | dispositivo: CPU | semilla: {SEED}")
"""),
    md("""
## 2. Dataset y significado de las variables

Se usa la version incluida en scikit-learn. Contiene **20,640 observaciones**, **8 features numericas** y un objetivo continuo. `MedHouseVal` se expresa en centenas de miles de dolares.

| Variable | Significado |
|---|---|
| `MedInc` | Ingreso mediano del grupo censal. |
| `HouseAge` | Edad mediana de las viviendas. |
| `AveRooms` | Promedio de habitaciones por hogar. |
| `AveBedrms` | Promedio de dormitorios por hogar. |
| `Population` | Poblacion del grupo censal. |
| `AveOccup` | Ocupacion promedio por hogar. |
| `Latitude` | Latitud del grupo censal. |
| `Longitude` | Longitud del grupo censal. |
| `MedHouseVal` | Target: valor mediano de vivienda en unidades de USD 100,000. |

Fuente: [documentacion oficial de California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
"""),
    code("""
housing = load_housing()
df = housing.frame.copy()
report = quality_report(df)
print(housing.DESCR.splitlines()[0])
display(df.head())
display(df.describe().T)
report
"""),
    md("""
### Calidad y decisiones de limpieza

- No existen valores nulos ni filas duplicadas, por lo que no se imputa ni elimina información.
- Todas las columnas son numericas; no existen categoricas y, por tanto, no se aplica one-hot encoding.
- La regla IQR marca valores extremos en varias variables. Se conservan porque pueden representar distritos reales y eliminarlos introduciria una decision arbitraria.
- El objetivo tiene un techo en `5.00001`: 965 observaciones alcanzan ese valor. Esto limita lo que el modelo puede aprender sobre viviendas por encima del umbral y se reconoce como limitacion.
- Las entradas se estandarizan porque sus escalas son muy distintas. El scaler se ajusta solo con entrenamiento para evitar fuga de datos.
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[TARGET].hist(bins=40, ax=axes[0], color="#2563eb", edgecolor="white")
axes[0].set(title="Distribucion del objetivo", xlabel="MedHouseVal (USD 100,000)", ylabel="Frecuencia")
axes[1].boxplot([df[col] for col in housing.feature_names], tick_labels=housing.feature_names, showfliers=True)
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_title("Boxplots de las variables predictoras")
fig.tight_layout()
fig.savefig(FIGURES / "eda_distribuciones.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
corr = df.corr(numeric_only=True)
image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
fig.colorbar(image, ax=ax, shrink=0.8, label="Correlacion de Pearson")
ax.set_title("Matriz de correlaciones")
fig.tight_layout()
fig.savefig(FIGURES / "eda_correlaciones.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
    md("""
## 3. Separacion y escalado sin fuga

Primero se reserva 30% y luego se divide ese bloque por la mitad. El resultado es 70% entrenamiento, 15% validacion y 15% test. Validacion sirve para comparar hiperparametros; test no participa en ninguna decision.
"""),
    code("""
data = split_and_scale(df)
validate_bundle(data)
split_summary = pd.DataFrame({
    "conjunto": ["Entrenamiento", "Validacion", "Test"],
    "observaciones": [len(data.X_train), len(data.X_val), len(data.X_test)],
    "porcentaje": [70, 15, 15],
})
display(split_summary)
print("Media escalada de train:", np.round(data.X_train_scaled.mean(axis=0), 6))
print("Desviacion escalada de train:", np.round(data.X_train_scaled.std(axis=0), 6))
"""),
    md("""
## 4. Investigacion de componentes de PyTorch

Las siguientes secciones incorporaran la investigacion de capas, funciones de perdida y optimizadores, seguida del modelo configurable y los 17 experimentos.
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, OUT)
print(f"Notebook creado: {OUT}")
