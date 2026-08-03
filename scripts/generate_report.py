"""Genera el reporte final de exactamente tres paginas con ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Laboratorio_1_Ian_Cumes_23236.pdf"
RESULTS = pd.read_csv(ROOT / "artifacts" / "experiments.csv")
SUMMARY = json.loads((ROOT / "artifacts" / "final_summary.json").read_text(encoding="utf-8"))
CURVES = ROOT / "artifacts" / "figures" / "curvas_seleccionadas.png"

PAGE_W, PAGE_H = landscape(letter)
MARGIN = 28
NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#2563eb")
CYAN = colors.HexColor("#0891b2")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#f1f5f9")
PALE_BLUE = colors.HexColor("#eff6ff")
GREEN = colors.HexColor("#15803d")
RED = colors.HexColor("#b91c1c")
WHITE = colors.white


BODY = ParagraphStyle(
    "Body", fontName="Helvetica", fontSize=7.2, leading=9.0, textColor=NAVY
)
SMALL = ParagraphStyle(
    "Small", fontName="Helvetica", fontSize=6.2, leading=7.5, textColor=NAVY
)
TINY = ParagraphStyle(
    "Tiny", fontName="Helvetica", fontSize=5.6, leading=6.7, textColor=NAVY
)
TABLE_HEAD = ParagraphStyle(
    "TableHead",
    fontName="Helvetica-Bold",
    fontSize=5.6,
    leading=6.5,
    textColor=WHITE,
    alignment=TA_CENTER,
)
CENTER = ParagraphStyle(
    "Center", fontName="Helvetica", fontSize=6.0, leading=7.0, alignment=TA_CENTER
)


def para(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def draw_title(c: canvas.Canvas, title: str, subtitle: str, page: int) -> float:
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 70, PAGE_W, 70, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(MARGIN, PAGE_H - 31, title)
    c.setFont("Helvetica", 8.5)
    c.drawString(MARGIN, PAGE_H - 49, subtitle)
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 48, f"PAGINA {page} / 3")
    return PAGE_H - 84


def draw_section_label(c: canvas.Canvas, x: float, y: float, text: str, width: float) -> float:
    c.setFillColor(BLUE)
    c.roundRect(x, y - 17, width, 17, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 7, y - 12, text.upper())
    return y - 23


def draw_paragraph(c: canvas.Canvas, text: str, x: float, y_top: float, width: float, style=BODY) -> float:
    p = para(text, style)
    _, height = p.wrap(width, PAGE_H)
    p.drawOn(c, x, y_top - height)
    return y_top - height


def draw_table(c: canvas.Canvas, data, x, y_top, widths, style, row_heights=None) -> float:
    table = Table(data, colWidths=widths, rowHeights=row_heights, repeatRows=1)
    table.setStyle(style)
    _, height = table.wrap(sum(widths), PAGE_H)
    table.drawOn(c, x, y_top - height)
    return y_top - height


def footer(c: canvas.Canvas) -> None:
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.line(MARGIN, 20, PAGE_W - MARGIN, 20)
    c.setFillColor(SLATE)
    c.setFont("Helvetica", 6.5)
    c.drawString(MARGIN, 9, "CC3092 - Deep Learning y Sistemas Inteligentes | Ian Cumes - Carne 23236")
    c.drawRightString(PAGE_W - MARGIN, 9, "California Housing | PyTorch | Semilla 42")


def page_one(c: canvas.Canvas) -> None:
    y = draw_title(
        c,
        "LABORATORIO 1 | ENTRENAMIENTO DE REDES NEURONALES",
        "MLP de regresion para California Housing - investigacion, metodologia y decisiones",
        1,
    )
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, "Ian Cumes | Carne 23236")
    repo = "https://github.com/iancumes/Lab1DeepLearning"
    c.setFillColor(BLUE)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN + 170, y, repo)
    c.linkURL(repo, (MARGIN + 168, y - 2, MARGIN + 430, y + 9), relative=0)
    y -= 16

    left_x, right_x = MARGIN, PAGE_W / 2 + 5
    col_w = PAGE_W / 2 - MARGIN - 12
    left_y = draw_section_label(c, left_x, y, "Capas y activaciones", col_w)
    layer_rows = [
        [para("<b>Componente</b>", TABLE_HEAD), para("Proposito y parametros", TABLE_HEAD)],
        [para("<b>nn.Linear</b>", SMALL), para("Transformacion afin y=xA^T+b. <b>in_features</b>, <b>out_features</b> y <b>bias</b> controlan dimensiones y termino independiente.", SMALL)],
        [para("<b>nn.ReLU</b>", SMALL), para("max(0,x): simple y eficiente; puede dejar neuronas inactivas si sus entradas permanecen negativas.", SMALL)],
        [para("<b>nn.LeakyReLU</b>", SMALL), para("Mantiene pendiente <b>negative_slope</b>=0.01 en x&lt;0 y conserva flujo de gradiente.", SMALL)],
        [para("<b>nn.Tanh</b>", SMALL), para("Salida centrada en [-1,1]; puede saturarse para magnitudes altas y reducir el gradiente.", SMALL)],
        [para("<b>nn.Dropout</b>", SMALL), para("Anula activaciones con probabilidad <b>p</b> durante train; se desactiva en evaluacion y reduce coadaptacion.", SMALL)],
        [para("<b>nn.BatchNorm1d</b>", SMALL), para("Normaliza cada feature del minibatch. <b>num_features</b>, <b>eps</b> y <b>momentum</b> controlan forma y estadisticas moviles.", SMALL)],
    ]
    common_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
    ])
    left_y = draw_table(c, layer_rows, left_x, left_y, [88, col_w - 88], common_style)

    right_y = draw_section_label(c, right_x, y, "Perdidas y optimizadores", col_w)
    loss_rows = [
        [para("<b>Elemento</b>", TABLE_HEAD), para("Funcion y diferencia", TABLE_HEAD)],
        [para("<b>MSELoss</b>", SMALL), para("Promedia errores cuadrados; enfatiza errores grandes. Objetivo principal.", SMALL)],
        [para("<b>L1Loss</b>", SMALL), para("Promedia errores absolutos; mas robusta ante extremos, no suave en cero.", SMALL)],
        [para("<b>SmoothL1Loss</b>", SMALL), para("Cuadratica cerca de cero y lineal lejos; <b>beta</b> fija la transicion.", SMALL)],
        [para("<b>SGD</b>", SMALL), para("Paso opuesto al gradiente; <b>momentum</b> suaviza oscilaciones. Exige ajustar lr.", SMALL)],
        [para("<b>Adam</b>", SMALL), para("Momentos adaptativos por parametro (<b>betas</b>, <b>eps</b>); convergencia inicial rapida.", SMALL)],
        [para("<b>RMSprop</b>", SMALL), para("Escala por media movil del gradiente cuadrado; <b>alpha</b> controla memoria.", SMALL)],
    ]
    right_y = draw_table(c, loss_rows, right_x, right_y, [92, col_w - 92], common_style)
    right_y -= 9
    right_y = draw_paragraph(
        c,
        "<b>lr:</b> tamano del paso; alto puede oscilar y bajo converge lentamente. "
        "<b>weight_decay:</b> penalizacion L2 aplicada por el optimizador. "
        "<b>L1:</b> se suma explicitamente como lambda por la suma de valores absolutos.",
        right_x,
        right_y,
        col_w,
        SMALL,
    )

    box_y = min(left_y, right_y) - 14
    box_h = 108
    c.setFillColor(PALE_BLUE)
    c.roundRect(MARGIN, box_y - box_h, PAGE_W - 2 * MARGIN, box_h, 6, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MARGIN + 9, box_y - 15, "DATOS, PREPARACION Y CONTROL DE FUGA")
    dataset_text = (
        "<b>Dataset:</b> 20,640 distritos, 8 features numericas y target MedHouseVal en USD 100,000. "
        "No hay nulos, duplicados ni variables categoricas. La regla IQR detecta extremos que se conservan por ser plausibles. "
        "El techo 5.00001 contiene 965 observaciones y limita la informacion sobre viviendas de mayor precio.<br/>"
        "<b>Split:</b> 70/15/15 = 14,448 train, 3,096 validacion y 3,096 test. StandardScaler se ajusta solo con train. "
        "Validacion decide hiperparametros; test se consulta una vez despues de cerrar la seleccion.<br/>"
        "<b>MLP:</b> Linear - BatchNorm opcional - activacion - Dropout opcional; salida Linear(1) sin activacion. "
        "CPU, semilla 42, algoritmos deterministas y checkpoint de menor RMSE de validacion."
    )
    draw_paragraph(c, dataset_text, MARGIN + 9, box_y - 24, PAGE_W - 2 * MARGIN - 18, BODY)
    c.setFillColor(SLATE)
    c.setFont("Helvetica-Oblique", 6.4)
    c.drawString(MARGIN + 9, box_y - box_h + 8, "Fuentes: scikit-learn California Housing; documentacion oficial de torch.nn y torch.optim.")
    footer(c)
    c.showPage()


def page_two(c: canvas.Canvas) -> None:
    y = draw_title(
        c,
        "BUSQUEDA CONTROLADA DE HIPERPARAMETROS",
        "17 corridas | metricas exclusivamente de validacion",
        2,
    )
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 7.2)
    c.drawString(MARGIN, y, "Baseline: [64,32] | ReLU | Adam lr=0.001 | batch=64 | 100 epochs | MSE | sin regularizacion")
    y -= 10

    headers = ["E", "Variacion", "Arq./Act.", "Opt./LR", "B/E", "Regularizacion", "Best", "MSE", "MAE", "RMSE", "s"]
    rows = [[para(h, TABLE_HEAD) for h in headers]]
    for _, row in RESULTS.sort_values("id").iterrows():
        reg = str(row["regularization"]).replace("Ninguna", "-")
        rows.append([
            para(str(int(row.id)), CENTER),
            para(str(row["name"]), TINY),
            para(f"{row['architecture']}<br/>{row['activation']}", CENTER),
            para(f"{row['optimizer']}<br/>{row['learning_rate']:.4g}", CENTER),
            para(f"{int(row['batch_size'])}<br/>{int(row['epochs'])}", CENTER),
            para(reg, TINY),
            para(str(int(row["best_epoch"])), CENTER),
            para(f"{row['val_mse']:.4f}", CENTER),
            para(f"{row['val_mae']:.4f}", CENTER),
            para(f"{row['val_rmse']:.4f}", CENTER),
            para(f"{row['training_seconds']:.1f}", CENTER),
        ])
    widths = [20, 96, 67, 61, 42, 70, 35, 48, 48, 48, 34]
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("BACKGROUND", (0, 8), (-1, 8), colors.HexColor("#dcfce7")),
        ("TEXTCOLOR", (0, 8), (-1, 8), GREEN),
    ])
    y = draw_table(c, rows, MARGIN, y, widths, table_style, row_heights=[15] + [15] * 17)
    y -= 8

    image = Image(str(CURVES), width=265, height=169)
    image.drawOn(c, MARGIN, y - 169)
    best = RESULTS.sort_values(["val_rmse", "val_mae", "parameters", "training_seconds"]).iloc[0]
    worst = RESULTS.sort_values("val_rmse", ascending=False).iloc[0]
    base = RESULTS.loc[RESULTS.id == 1].iloc[0]
    l1 = RESULTS.loc[RESULTS.id == 14].iloc[0]
    l2 = RESULTS.loc[RESULTS.id == 15].iloc[0]
    dropout = RESULTS.loc[RESULTS.id == 16].iloc[0]
    x = MARGIN + 282
    w = PAGE_W - MARGIN - x
    y2 = draw_section_label(c, x, y, "Lectura de resultados", w)
    insights = (
        f"<b>Ganador:</b> E{int(best.id)} ({best['name']}), RMSE={best.val_rmse:.4f}; mejora "
        f"{base.val_rmse-best.val_rmse:.4f} frente al baseline. El lr=0.01 avanzo mas por actualizacion sin divergir.<br/><br/>"
        f"<b>Mayor impacto negativo:</b> E{int(worst.id)} ({worst['name']}), RMSE={worst.val_rmse:.4f}. "
        "El lr=0.0001 fue demasiado conservador para 100 epochs y produjo underfitting relativo.<br/><br/>"
        f"<b>Regularizacion:</b> L1={l1.val_rmse:.4f}, L2={l2.val_rmse:.4f}, Dropout={dropout.val_rmse:.4f}. "
        "L2 fue la mejor de estas tres, aunque la mejora frente al baseline fue pequena; Dropout perjudico este MLP compacto.<br/><br/>"
        "<b>Velocidad:</b> batch 256 fue mas rapido, pero menos preciso; batch 32 requirio mas actualizaciones. "
        "150 epochs mejoro respecto a 50, aunque el checkpoint optimo aparecio antes del final."
    )
    draw_paragraph(c, insights, x, y2, w, BODY)
    footer(c)
    c.showPage()


def metric_card(c, x, y, w, title, value, detail, color):
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.roundRect(x, y, w, 60, 6, fill=1, stroke=1)
    c.setFillColor(color)
    c.rect(x, y, 5, 60, fill=1, stroke=0)
    c.setFillColor(SLATE)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(x + 12, y + 45, title)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x + 12, y + 24, value)
    c.setFillColor(SLATE)
    c.setFont("Helvetica", 6.3)
    c.drawString(x + 12, y + 10, detail)


def page_three(c: canvas.Canvas) -> None:
    y = draw_title(
        c,
        "MODELO FINAL, DISCUSION Y CONCLUSIONES",
        "Seleccion cerrada con validacion; evaluacion de test realizada una sola vez",
        3,
    )
    test = SUMMARY["test"]
    best = RESULTS.loc[RESULTS.id == SUMMARY["best_experiment"]].iloc[0]
    card_y = y - 62
    gap = 9
    card_w = (PAGE_W - 2 * MARGIN - 3 * gap) / 4
    metric_card(c, MARGIN, card_y, card_w, "CONFIGURACION GANADORA", "E8 | lr=0.01", "[64,32] - ReLU - Adam - batch 64", BLUE)
    metric_card(c, MARGIN + card_w + gap, card_y, card_w, "TEST MSE", f"{test['mse']:.4f}", "unidades del target al cuadrado", CYAN)
    metric_card(c, MARGIN + 2 * (card_w + gap), card_y, card_w, "TEST MAE", f"{test['mae']:.4f}", f"aprox. USD {test['mae_usd']:,.0f}", GREEN)
    metric_card(c, MARGIN + 3 * (card_w + gap), card_y, card_w, "TEST RMSE", f"{test['rmse']:.4f}", f"aprox. USD {test['rmse_usd']:,.0f}", RED)

    y = card_y - 14
    left_x, right_x = MARGIN, PAGE_W / 2 + 5
    col_w = PAGE_W / 2 - MARGIN - 12
    left_y = draw_section_label(c, left_x, y, "Discusion 1-3", col_w)
    left_text = (
        f"<b>1. Impacto.</b> El cambio positivo mayor fue elevar lr de 0.001 a 0.01: RMSE bajo de "
        f"{RESULTS.loc[RESULTS.id==1,'val_rmse'].iloc[0]:.4f} a {best.val_rmse:.4f}. El peor cambio fue lr=0.0001 "
        f"(RMSE {RESULTS.loc[RESULTS.id==9,'val_rmse'].iloc[0]:.4f}), porque avanzo muy poco en 100 epochs.<br/><br/>"
        "<b>2. Overfitting/underfitting.</b> Las curvas y la brecha val-train identifican sobreajuste cuando train sigue bajando "
        "pero validacion deja de mejorar. El checkpoint por mejor epoch evita conservar una etapa posterior peor. El lr bajo mostro "
        "underfitting relativo por convergencia insuficiente.<br/><br/>"
        f"<b>3. Regularizacion.</b> L2 fue la mejor entre L1/L2/Dropout (RMSE {RESULTS.loc[RESULTS.id==15,'val_rmse'].iloc[0]:.4f}), "
        "pero la ganancia fue marginal. Dropout elimino capacidad util en una red pequena y elevo el error."
    )
    left_y = draw_paragraph(c, left_text, left_x, left_y, col_w, BODY)

    right_y = draw_section_label(c, right_x, y, "Discusion 4-6", col_w)
    right_text = (
        f"<b>4. Batch y epochs.</b> Batch 256 tardo {RESULTS.loc[RESULTS.id==11,'training_seconds'].iloc[0]:.1f}s frente a "
        f"{RESULTS.loc[RESULTS.id==10,'training_seconds'].iloc[0]:.1f}s con batch 32, pero tuvo peor RMSE. Mas epochs ofrecen "
        "oportunidad de converger, no garantia de generalizar; el mejor checkpoint puede ocurrir antes.<br/><br/>"
        "<b>5. Metricas.</b> MSE enfatiza errores grandes y queda en unidades cuadradas; MAE expresa el error absoluto tipico y "
        "resiste mejor outliers; RMSE conserva la penalizacion cuadratica en las unidades originales.<br/><br/>"
        "<b>6. Produccion.</b> Se elegiria E8 como punto de partida y busqueda bayesiana con validacion cruzada. Tambien se "
        "monitorearian drift, error por region y segmento, latencia y el efecto del target censurado antes del despliegue."
    )
    right_y = draw_paragraph(c, right_text, right_x, right_y, col_w, BODY)

    y = min(left_y, right_y) - 13
    y = draw_section_label(c, MARGIN, y, "Limitaciones, conclusion y referencias", PAGE_W - 2 * MARGIN)
    final_text = (
        "<b>Limitaciones.</b> El techo del target impide conocer el precio real de 965 distritos; un split aleatorio puede colocar zonas "
        "cercanas en conjuntos distintos y ser optimista respecto a generalizacion geografica. Una sola semilla no cuantifica variabilidad. "
        "El modelo aprende asociaciones, no causalidad.<br/>"
        f"<b>Conclusion.</b> El MLP final generalizo con RMSE de USD {test['rmse_usd']:,.0f} y MAE de USD {test['mae_usd']:,.0f}. "
        "La tasa de aprendizaje tuvo el mayor efecto: un paso mayor mejoro convergencia, mientras uno bajo dejo el modelo corto de entrenamiento. "
        "Regularizacion y complejidad deben decidirse con validacion, no por intuicion ni observando test.<br/>"
        "<b>Referencias:</b> scikit-learn, California Housing dataset; PyTorch, torch.nn API y torch.optim API. "
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html | "
        "https://docs.pytorch.org/docs/stable/nn.html | https://docs.pytorch.org/docs/stable/optim.html"
    )
    draw_paragraph(c, final_text, MARGIN, y, PAGE_W - 2 * MARGIN, SMALL)
    footer(c)
    c.showPage()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=landscape(letter), pageCompression=1)
    c.setTitle("Laboratorio 1 - Entrenamiento de redes neuronales")
    c.setAuthor("Ian Cumes - Carne 23236")
    c.setSubject("MLP de regresion para California Housing")
    page_one(c)
    page_two(c)
    page_three(c)
    c.save()
    print(f"PDF creado: {OUTPUT}")


if __name__ == "__main__":
    main()
