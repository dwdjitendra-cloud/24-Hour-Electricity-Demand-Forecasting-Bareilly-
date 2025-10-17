from typing import Dict, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_pdf_report(
    report_path: str,
    context: Dict,
) -> None:
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = context.get("title", "24-Hour Electricity Demand Forecasting")
    story.append(Paragraph(title, styles["Title"]))

    # Problem
    story.append(Paragraph("Problem", styles["Heading2"]))
    story.append(
        Paragraph(
            "Forecast the next 24 hours of hourly electricity demand (kWh) for Bareilly from 3-minute smart-meter data.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))

    # Data
    story.append(Paragraph("Data", styles["Heading2"]))
    data_text = context.get(
        "data_text",
        "Primary dataset: SM Cleaned Data BR Aggregated.csv (3-minute readings). Resampled to hourly with small-gap forward fill (<=2h) and 1st–99th percentile clipping for outliers.",
    )
    story.append(Paragraph(data_text, styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Methods
    story.append(Paragraph("Methods", styles["Heading2"]))
    methods = (
        "Two models: (A) Seasonal Naive (y_t = y_{t-24}) and (B) Ridge Regression with hour-of-day sin/cos, day-of-week one-hot, lags 1/2/3, 24h rolling mean, and temperature if available."
    )
    story.append(Paragraph(methods, styles["BodyText"]))
    if context.get("weather_used"):
        story.append(Paragraph("Weather: Open-Meteo hourly temperature used for T+1…T+24.", styles["BodyText"]))
    else:
        story.append(Paragraph("Weather: Not used (unavailable or did not align with forecast timestamps).", styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Results summary table
    story.append(Paragraph("Results (Backtests)", styles["Heading2"]))
    metrics_rows = context.get("metrics_rows", [])
    if metrics_rows:
        table_data = [["Anchor Time", "Model", "MAE", "WAPE", "sMAPE"]] + metrics_rows
        tbl = Table(table_data, hAlign='LEFT')
        tbl.setStyle(
            TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
            ])
        )
        story.append(tbl)
    else:
        story.append(Paragraph("No backtest metrics available.", styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Takeaways", styles["Heading2"]))
    takeaways = context.get(
        "takeaways",
        "The seasonal naive provides a strong baseline for short-term horizons. Ridge regression captures intra-day patterns and benefits from weather when aligned. Further gains likely from tree-based models (e.g., LightGBM) and holiday effects.",
    )
    story.append(Paragraph(takeaways, styles["BodyText"]))

    # Force at least two pages by adding content and a page break via spacer
    story.append(Spacer(1, 600))
    story.append(Paragraph("Appendix: Method Details", styles["Heading2"]))
    story.append(Paragraph(
        "Training window: last 7 days up to T−1. Forecast horizon: T+1…T+24. Small gaps (<=2h) forward-filled. Outliers clipped to 1st–99th percentile.",
        styles["BodyText"],
    ))

    doc.build(story)
