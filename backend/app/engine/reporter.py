from datetime import datetime
from typing import Dict, Any, List, Optional
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def _signed(value: float) -> str:
    return f"+{value}" if value >= 0 else f"{value}"


def generate_pdf_report(
    dataset_name: str,
    summary_stats: Dict[str, Any],
    before_health: Dict[str, Any],
    after_health: Optional[Dict[str, Any]],
    transformations: List[str],
    benchmarks: List[Dict[str, Any]],
    output_pdf_path: str,
) -> str:
    """
    Generates a professional executive PDF Data Quality & ML Readiness Report.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15,
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155'),
    )
    badge_style = ParagraphStyle(
        'BadgeText',
        parent=styles['Normal'],
        fontSize=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor('#0284c7'),
    )

    story = []
    dataset_name = escape(dataset_name)

    # Title & Subtitle
    story.append(Paragraph("AI Data Readiness Audit Report", title_style))
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b> | Generated on {datetime.utcnow().strftime('%B %d, %Y %H:%M UTC')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=15))

    # Section 1: Executive Summary & Health Scores
    story.append(Paragraph("1. Executive Data Health Summary", section_heading))
    before_score = before_health.get("composite_score", 0.0)
    executed = after_health is not None
    after_health = after_health or {}

    if executed:
        after_score = after_health.get("composite_score", before_score)
        score_delta = round(after_score - before_score, 1)
        summary_text = (
            f"The dataset was profiled and diagnosed for pre-machine learning readiness. "
            f"Initial Data Health Score was <b>{before_score}/100</b> ({before_health.get('grade', 'N/A')}). "
            f"After executing the user-approved preprocessing steps, the Data Health Score is "
            f"<b>{after_score}/100</b> ({after_health.get('grade', 'N/A')}), a change of <b>{_signed(score_delta)} pts</b>."
        )
    else:
        summary_text = (
            f"The dataset was profiled and diagnosed for pre-machine learning readiness. "
            f"Current Data Health Score is <b>{before_score}/100</b> ({before_health.get('grade', 'N/A')}). "
            f"No cleaning pipeline has been executed yet."
        )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))

    def sub(health: Dict[str, Any], key: str) -> str:
        val = (health.get("sub_scores") or {}).get(key)
        return f"{val}%" if val is not None else "-"

    def sub_delta(key: str) -> str:
        b = (before_health.get("sub_scores") or {}).get(key)
        a = (after_health.get("sub_scores") or {}).get(key)
        if a is None or b is None:
            return "-"
        return f"{_signed(round(a - b, 1))} pts"

    score_data = [
        ["Health Dimension", "Before Processing", "After Processing", "Change"],
        ["Composite Health Score", f"{before_score} / 100",
         f"{after_health.get('composite_score')} / 100" if executed else "-",
         f"{_signed(round(after_health['composite_score'] - before_score, 1))} pts" if executed else "-"],
    ]
    for label, key in [
        ("Missingness Sub-score", "missingness_score"),
        ("Duplicate Sub-score", "duplicate_score"),
        ("Outlier Sub-score", "outlier_score"),
        ("Validity Sub-score", "validity_score"),
        ("Target Balance Sub-score", "target_balance_score"),
        ("Feature Quality Sub-score", "feature_quality_score"),
    ]:
        score_data.append([label, sub(before_health, key), sub(after_health, key), sub_delta(key)])
    t1 = Table(score_data, colWidths=[160, 120, 120, 100])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    # Section 2: Transformations Applied
    story.append(Paragraph("2. Executed Preprocessing Pipeline", section_heading))
    if not executed:
        story.append(Paragraph("The cleaning pipeline has not been executed for this dataset yet.", body_style))
    elif transformations:
        for idx, step in enumerate(transformations):
            story.append(Paragraph(f"• <b>Step {idx + 1}:</b> {escape(str(step))}", body_style))
    else:
        story.append(Paragraph("The pipeline ran with no transformations approved.", body_style))
    story.append(Spacer(1, 15))

    # Section 3: Model Benchmark Recommendations
    story.append(Paragraph("3. Downstream Model Recommendations", section_heading))
    if benchmarks:
        story.append(Paragraph("Cross-validated model fits executed on the cleaned dataset:", body_style))
        story.append(Spacer(1, 8))
        bench_data = [["Rank", "Candidate Model", "Primary Metric", "Metric Value", "Fit Time", "Suitability"]]
        for b in benchmarks[:5]:
            bench_data.append([
                str(b.get("rank", "-")),
                b.get("model_name", ""),
                b.get("metric_name", ""),
                b.get("metric_display", ""),
                f"{b.get('training_time_sec', 0)}s",
                b.get("suitability", "-"),
            ])
        t2 = Table(bench_data, colWidths=[40, 160, 120, 80, 50, 60])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        story.append(t2)
    else:
        story.append(Paragraph("No model benchmarks have been run for this dataset yet.", body_style))
    story.append(Spacer(1, 20))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=10))
    story.append(Paragraph("Report automatically compiled by AI Data Readiness Platform. All recommendations are mathematically reproducible.", subtitle_style))

    doc.build(story)
    return output_pdf_path
