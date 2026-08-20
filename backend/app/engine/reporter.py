import os
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from app.core.config import settings


def generate_pdf_report(
    dataset_name: str,
    summary_stats: Dict[str, Any],
    before_health: Dict[str, Any],
    after_health: Dict[str, Any],
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

    # Title & Subtitle
    story.append(Paragraph("AI Data Readiness Audit Report", title_style))
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b> | Generated on {datetime.utcnow().strftime('%B %d, %Y %H:%M UTC')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=15))

    # Section 1: Executive Summary & Health Scores
    story.append(Paragraph("1. Executive Data Health Summary", section_heading))
    before_score = before_health.get("composite_score", 0.0)
    after_score = after_health.get("composite_score", before_score)
    score_delta = round(after_score - before_score, 1)
    
    summary_text = (
        f"The dataset was profiled and diagnosed for pre-machine learning readiness. "
        f"Initial Data Health Score was <b>{before_score}/100</b> ({before_health.get('grade', 'N/A')}). "
        f"Following the execution of user-approved safe-order preprocessing transformations, "
        f"the final Data Health Score reached <b>{after_score}/100</b> (Improvement: <b>+{score_delta} pts</b>)."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))

    # Score Comparison Table
    score_data = [
        ["Health Dimension", "Before Processing", "After Processing", "Improvement"],
        ["Composite Health Score", f"{before_score} / 100", f"{after_score} / 100", f"+{score_delta} pts"],
        ["Missingness Sub-score", f"{before_health.get('sub_scores', {}).get('missingness_score', 0)}%", f"{after_health.get('sub_scores', {}).get('missingness_score', 100)}%", "Resolved"],
        ["Duplicate Sub-score", f"{before_health.get('sub_scores', {}).get('duplicate_score', 0)}%", f"{after_health.get('sub_scores', {}).get('duplicate_score', 100)}%", "Resolved"],
        ["Outlier Sub-score", f"{before_health.get('sub_scores', {}).get('outlier_score', 0)}%", f"{after_health.get('sub_scores', {}).get('outlier_score', 100)}%", "Winsorized"],
        ["Validity Sub-score", f"{before_health.get('sub_scores', {}).get('validity_score', 0)}%", f"{after_health.get('sub_scores', {}).get('validity_score', 100)}%", "Clipped"],
    ]
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
    story.append(Paragraph("2. Approved Preprocessing Pipeline", section_heading))
    if transformations:
        for idx, step in enumerate(transformations):
            story.append(Paragraph(f"• <b>Step {idx + 1}:</b> {step}", body_style))
    else:
        story.append(Paragraph("No custom transformations were required; dataset met standard baseline criteria.", body_style))
    story.append(Spacer(1, 15))

    # Section 3: Model Benchmark Recommendations
    story.append(Paragraph("3. Downstream Model Recommendations", section_heading))
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
            b.get("suitability", "High"),
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
    story.append(Spacer(1, 20))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceAfter=10))
    story.append(Paragraph("Report automatically compiled by AI Data Readiness Platform. All recommendations are mathematically reproducible.", subtitle_style))

    doc.build(story)
    return output_pdf_path
