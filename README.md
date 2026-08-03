# Laboratorio 1 - Entrenamiento de redes neuronales

**Curso:** CC3092 - Deep Learning y Sistemas Inteligentes<br>
**Estudiante:** Ian Cumes<br>
**Carne:** 23236

Este repositorio contiene una implementacion reproducible de un perceptron multicapa (MLP) para regresion sobre California Housing. El laboratorio compara 17 configuraciones de arquitectura, activacion, optimizador, tasa de aprendizaje, batch size, epochs y regularizacion sin utilizar el conjunto de prueba durante la seleccion.

## Entorno

Se recomienda Python 3.11. En Windows, una ruta corta para el entorno evita el limite historico de longitud de rutas que pueden alcanzar los paquetes de PyTorch:

```powershell
py -3.11 -m venv C:\venvs\lab1dl
C:\venvs\lab1dl\Scripts\python -m pip install --upgrade pip
C:\venvs\lab1dl\Scripts\python -m pip install -r requirements.txt
```

## Ejecucion reproducible

Registre el entorno como kernel, construya el notebook y ejecutelo desde cero:

```powershell
C:\venvs\lab1dl\Scripts\python -m ipykernel install --user --name lab1dl --display-name "Lab1 Deep Learning"
C:\venvs\lab1dl\Scripts\python scripts\build_notebook.py
C:\venvs\lab1dl\Scripts\python scripts\execute_notebook.py
```

La ejecucion en CPU tarda aproximadamente 10 minutos en el equipo usado para la entrega. Descarga el dataset mediante `fetch_california_housing`, ejecuta las 17 configuraciones y evalua test solo despues de seleccionar la mejor corrida por validacion.

Para regenerar el reporte y validar todos los entregables:

```powershell
C:\venvs\lab1dl\Scripts\python scripts\generate_report.py
C:\venvs\lab1dl\Scripts\python scripts\validate_project.py
```

## Resultados principales

- Mejor configuracion de validacion: experimento 8, `[64, 32]`, ReLU, Adam, `lr=0.01`, batch 64.
- RMSE de validacion: `0.5282`.
- Evaluacion unica de test: MSE `0.2516`, MAE `0.3481` y RMSE `0.5016`.
- En dolares aproximados: MAE USD 34,814 y RMSE USD 50,156.

## Estructura

- `Lab1_Entrenamiento_MLP_California_Housing.ipynb`: entrega principal, ejecutada y comentada.
- `src/`: carga, split, escalado, MLP y entrenador reutilizable.
- `artifacts/`: resultados CSV, historiales JSON y figuras generadas.
- `scripts/`: construccion, ejecucion, reporte y validacion automatizada.
- `output/pdf/Laboratorio_1_Ian_Cumes_23236.pdf`: reporte final de tres paginas.
- `Laboratorio #1.md`: enunciado original.

El cache del dataset, los entornos virtuales y los archivos temporales se excluyen de Git.
