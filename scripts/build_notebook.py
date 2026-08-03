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

### Capas y activaciones

| Componente | Proposito y parametros relevantes |
|---|---|
| `nn.Linear(in_features, out_features, bias)` | Transformacion afin `y=xA^T+b`; conecta cada neurona con la capa siguiente. `in_features` y `out_features` definen dimensiones y `bias` controla el termino independiente. |
| `nn.ReLU(inplace)` | Aplica `max(0,x)`. Es barata y reduce el problema de gradientes que se desvanecen, aunque una neurona puede quedar inactiva. |
| `nn.LeakyReLU(negative_slope, inplace)` | Conserva una pendiente pequena para valores negativos; `negative_slope=0.01` evita gradiente exactamente cero. |
| `nn.Tanh()` | Comprime a `[-1,1]` y centra las activaciones, pero puede saturarse y reducir el gradiente. |
| `nn.Dropout(p)` | Durante entrenamiento anula aleatoriamente una fraccion `p`; en evaluacion se desactiva. Reduce coadaptacion y overfitting. |
| `nn.BatchNorm1d(num_features, eps, momentum)` | Normaliza activaciones por feature en minibatches y mantiene estadisticas moviles para evaluacion. Puede estabilizar y acelerar entrenamiento. |

Fuente: [API oficial de torch.nn](https://docs.pytorch.org/docs/stable/nn.html).

### Perdidas de regresion

| Perdida | Interpretacion |
|---|---|
| `nn.MSELoss(reduction)` | Promedia errores al cuadrado. Penaliza con fuerza errores grandes y es la funcion objetivo principal del laboratorio. |
| `nn.L1Loss(reduction)` | Promedia errores absolutos. Es mas robusta ante valores extremos, pero su gradiente no es suave en cero. |
| `nn.SmoothL1Loss(beta, reduction)` | Combina comportamiento cuadratico cerca de cero y lineal para errores grandes; `beta` controla la transicion. |

Las tres aceptan `reduction='mean'`, `'sum'` o `'none'`. MSE de entrenamiento no debe confundirse con RMSE: RMSE es la raiz de MSE y vuelve a las unidades del target.

### Optimizadores y regularizacion

| Optimizador | Funcion y diferencia principal |
|---|---|
| `SGD(params, lr, momentum, weight_decay)` | Actualiza en direccion contraria al gradiente. Momentum acumula direccion y reduce oscilaciones; suele requerir mayor ajuste de `lr`. |
| `Adam(params, lr, betas, eps, weight_decay)` | Adapta el paso por parametro usando momentos de primer y segundo orden. Suele converger rapido y es el baseline. |
| `RMSprop(params, lr, alpha, eps, weight_decay)` | Divide el gradiente por una media movil de gradientes cuadrados; `alpha` controla la memoria del acumulador. |

`lr` determina el tamano del paso: demasiado alto puede oscilar o divergir y demasiado bajo vuelve lento el aprendizaje. `weight_decay` agrega regularizacion L2 en estos optimizadores. La regularizacion L1 se incorpora explicitamente a la funcion objetivo. Fuente: [API oficial de torch.optim](https://docs.pytorch.org/docs/stable/optim.html).
"""),
    md("""
## 5. MLP configurable y estrategia experimental

Cada bloque oculto sigue `Linear -> BatchNorm opcional -> activacion -> Dropout opcional`; la salida es una capa lineal de una neurona, apropiada para regresion sin restringir el rango. Todas las corridas reinician la semilla, usan el mismo split y conservan el checkpoint con menor RMSE de validacion.
"""),
    code("""
from dataclasses import asdict
from IPython.display import Markdown, display
from src.experiments import (
    RegressionMLP, experiment_catalog, plot_selected_curves,
    retrain_and_test_once, run_experiments, save_artifacts, select_best,
)

configs = experiment_catalog()
config_table = pd.DataFrame([asdict(config) for config in configs])
assert len(config_table) == 17 and config_table["id"].is_unique
display(config_table)

shape_test = RegressionMLP(8, configs[0])(torch.zeros(5, 8))
assert tuple(shape_test.shape) == (5,)
print("Forma de salida validada:", tuple(shape_test.shape))
"""),
    md("""
## 6. Entrenamiento de las 17 iteraciones

La perdida comparable entre train y validacion es MSE. En L1 tambien se registra `train_objective`, que suma la penalizacion a MSE. El conjunto de test no se pasa a `run_experiments` y permanece aislado.
"""),
    code("""
results, histories, states = run_experiments(
    configs,
    data.X_train_scaled,
    data.y_train.to_numpy(dtype=np.float32),
    data.X_val_scaled,
    data.y_val.to_numpy(dtype=np.float32),
)

summary_columns = [
    "id", "name", "architecture", "activation", "optimizer", "learning_rate",
    "batch_size", "epochs", "regularization", "best_epoch", "val_mse",
    "val_mae", "val_rmse", "training_seconds",
]
display(results[summary_columns].round(5))
"""),
    md("""
## 7. Curvas, seleccion y diagnostico

Se muestran baseline, mejor, peor y la corrida con mayor brecha MSE de validacion-entrenamiento. Una brecha creciente sugiere overfitting; perdidas altas y cercanas sugieren underfitting.
"""),
    code("""
best = select_best(results)
curve_ids = plot_selected_curves(results, histories, FIGURES / "curvas_seleccionadas.png")
plt.show()
display(best[summary_columns].to_frame("mejor_configuracion"))
print("Curvas graficadas:", curve_ids)
"""),
    md("""
## 8. Evaluacion final sobre test - una sola vez

Solo despues de fijar la configuracion ganadora se combinan train y validacion, se reajusta el scaler con ese conjunto y se reentrena durante el mejor epoch observado. Entonces se evalua test una unica vez.
"""),
    code("""
best_config = next(config for config in configs if config.id == int(best["id"]))
X_train_val = pd.concat([data.X_train, data.X_val]).sort_index()
y_train_val = pd.concat([data.y_train, data.y_val]).sort_index()

final_metrics, final_scaler, final_state = retrain_and_test_once(
    best_config,
    int(best["best_epoch"]),
    X_train_val,
    y_train_val,
    data.X_test,
    data.y_test,
)
save_artifacts(ARTIFACTS, results, histories, best, final_metrics)

display(pd.DataFrame({
    "metrica": ["MSE", "MAE", "RMSE", "MAE aproximado (USD)", "RMSE aproximado (USD)"],
    "test": [final_metrics["mse"], final_metrics["mae"], final_metrics["rmse"], final_metrics["mae_usd"], final_metrics["rmse_usd"]],
}))
"""),
    md("""
## 9. Discusion y conclusiones basadas en resultados

La siguiente celda produce una sintesis usando exclusivamente validacion para comparar configuraciones y reserva test para estimar generalizacion final.
"""),
    code("""
baseline = results.loc[results.id == 1].iloc[0]
best_regularized = results.loc[results.regularization != "Ninguna"].sort_values("val_rmse").iloc[0]
worst = results.sort_values("val_rmse", ascending=False).iloc[0]
largest_gap = results.sort_values("generalization_gap", ascending=False).iloc[0]
batch_rows = results.loc[results.id.isin([10, 11])].sort_values("batch_size")
epoch_rows = results.loc[results.id.isin([12, 13])].sort_values("epochs")

discussion = f'''
### Hallazgos

1. **Mayor impacto positivo y negativo.** La mejor validacion fue E{int(best.id)} ({best['name']}) con RMSE {best.val_rmse:.4f}, una variacion de {(best.val_rmse-baseline.val_rmse):+.4f} frente al baseline. El peor resultado fue E{int(worst.id)} ({worst['name']}) con RMSE {worst.val_rmse:.4f}.
2. **Overfitting/underfitting.** La mayor brecha en su mejor epoch aparecio en E{int(largest_gap.id)} ({largest_gap['name']}), con `val_mse-train_mse={largest_gap.generalization_gap:.4f}`. Las curvas permiten distinguir brecha creciente (overfitting) de perdidas altas y similares (underfitting).
3. **Regularizacion.** La mejor variante regularizada fue E{int(best_regularized.id)} ({best_regularized.regularization}), RMSE {best_regularized.val_rmse:.4f}; frente al baseline el cambio fue {(best_regularized.val_rmse-baseline.val_rmse):+.4f}. Dropout introduce ruido, L1 favorece pesos pequenos o cero y L2 penaliza suavemente magnitudes grandes.
4. **Batch size y epochs.** Batch 32 y 256 tardaron {batch_rows.iloc[0].training_seconds:.1f}s y {batch_rows.iloc[1].training_seconds:.1f}s, respectivamente. Batches pequenos generan mas actualizaciones y ruido; grandes producen gradientes mas estables. Las corridas de 50 y 150 epochs muestran el intercambio entre tiempo, convergencia y posible sobreajuste.
5. **MSE, MAE y RMSE.** MSE amplifica errores grandes y queda en unidades cuadradas; MAE describe el error absoluto tipico y es mas robusta; RMSE conserva la penalizacion cuadratica pero vuelve a unidades de USD 100,000. En test: MSE={final_metrics['mse']:.4f}, MAE={final_metrics['mae']:.4f} y RMSE={final_metrics['rmse']:.4f}.
6. **Eleccion de produccion.** Se elegiria la arquitectura y los hiperparametros de E{int(best.id)}, acompañados de validacion cruzada o busqueda bayesiana, monitoreo de drift, analisis geoespacial del error y una estrategia explicita para el target censurado.

### Conclusion

La comparacion controlada muestra que el rendimiento no depende solo del tamano de la red: escala de entradas, optimizador, tasa de aprendizaje, regularizacion y tiempo de entrenamiento interactuan. El test final se consulto una vez y obtuvo RMSE aproximado de USD {final_metrics['rmse_usd']:,.0f} y MAE aproximado de USD {final_metrics['mae_usd']:,.0f}. Estos valores deben interpretarse considerando el techo de `MedHouseVal` y que un split aleatorio no evalua por completo generalizacion geografica o temporal.
'''
display(Markdown(discussion))
"""),
    md("""
## Referencias

- Scikit-learn. *California housing dataset*. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- PyTorch. *torch.nn API*. https://docs.pytorch.org/docs/stable/nn.html
- PyTorch. *torch.optim API*. https://docs.pytorch.org/docs/stable/optim.html
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
