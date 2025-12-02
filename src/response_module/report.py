# -*- coding: utf-8 -*- noqa
"""Report generation helper for response_module.

Provides a single function `generate_report(patient_id, result, out_dir='reports')`
that builds and returns the path to a PDF report.
"""

import environment
import os
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_report(patient_id: str, result: dict, out_dir: str = 'reports') -> str:
    """Generate a PDF report for a patient and return the filename.

    Raises exceptions on failure so callers can handle HTTP responses.
    """
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime(environment.time_format)
    pdf_filename = os.path.join(out_dir, f'Report CKD {patient_id} {timestamp}.pdf')

    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
    )

    story.append(Paragraph('REPORTE DE EVALUACIÓN CKD', title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(f'<b>Paciente ID:</b> {patient_id}', styles['Normal']))
    story.append(Paragraph(f'<b>Fecha:</b> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph('RESULTADO DE PREDICCIÓN', subtitle_style))

    prob = result.get('probability', 0)
    diagnosis = result.get('diagnosis', 'N/A')
    confidence = result.get('confidence', 0)

    diag_color = colors.HexColor('#E74C3C') if diagnosis == 'CKD' else colors.HexColor('#27AE60')

    main_data = [
        ['Métrica', 'Valor'],
        ['Probabilidad de CKD', f'{prob:.4f}'],
        ['Diagnóstico', diagnosis],
        ['Confianza', f'{confidence:.2%}'],
        ['Distancia al threshold', f"{result.get('distance_to_threshold', 0):.4f}"],
        ['Threshold usado', f"{result.get('threshold_used', 0):.4f} ({result.get('threshold_type', 'N/A').upper()})"],
    ]

    main_table = Table(main_data, colWidths=[3 * inch, 3 * inch])
    main_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 2), (0, 2), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1, 2), (1, 2), diag_color),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
    ]))

    story.append(main_table)
    story.append(Spacer(1, 0.3 * inch))

    interpretation = result.get('interpretation', '')
    if interpretation:
        story.append(Paragraph('INTERPRETACIÓN', subtitle_style))
        story.append(Paragraph(interpretation, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

    top_features = result.get('top_features', [])
    available_features = [f for f in top_features if f.get('value') is not None and f.get('value') != -1]

    if available_features:
        story.append(Paragraph('CARACTERÍSTICAS MÁS RELEVANTES (Top 10)', subtitle_style))

        features_data = [['#', 'Característica', 'Valor', 'Importancia']]
        for i, feature in enumerate(available_features[:10], 1):
            importance = feature.get('importance', 0)
            importance_bar = '█' * int(importance * 20)
            value = feature.get('value', 'N/A')
            if isinstance(value, (int, float)):
                value_str = f'{value:.4f}' if isinstance(value, float) else str(value)
            else:
                value_str = str(value)

            features_data.append([
                str(i),
                feature.get('feature', 'N/A'),
                value_str,
                f'{importance_bar} {importance:.3f}'
            ])

        features_table = Table(features_data, colWidths=[0.5 * inch, 2.5 * inch, 1.5 * inch, 2 * inch])
        features_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        story.append(features_table)
        story.append(Spacer(1, 0.3 * inch))

        all_features_data = [['#', 'Característica', 'Valor']]
        for i, feature in enumerate(available_features, 1):
            value = feature.get('value', 'N/A')
            if isinstance(value, (int, float)):
                value_str = f'{value:.4f}' if isinstance(value, float) else str(value)
            else:
                value_str = str(value)

            all_features_data.append([str(i), feature.get('feature', 'N/A'), value_str])

        all_features_table = Table(all_features_data, colWidths=[0.5 * inch, 3.5 * inch, 2.5 * inch])
        all_features_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16A085')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))

        story.append(all_features_table)
        story.append(Spacer(1, 0.2 * inch))

        completeness = len(available_features) / len(top_features) if top_features else 0
        if completeness < 0.5:
            warning_style = ParagraphStyle('Warning', parent=styles['Normal'], textColor=colors.HexColor('#E67E22'), fontSize=10, leading=14)
            total_possible = len(top_features)
            available_count = len(available_features)

            story.append(Paragraph(
                f"<b>ATENCIÓN:</b> Solo {available_count}/{total_possible} características principales están disponibles. El modelo puede tener en cuenta hasta {total_possible} características para una evaluación más precisa.",
                warning_style
            ))
            story.append(Spacer(1, 0.1 * inch))
            recommendation_style = ParagraphStyle('Recommendation', parent=styles['Normal'], fontSize=10, leading=14, leftIndent=20)
            story.append(Paragraph('<b>Para un análisis más preciso siempre recomendamos completar:</b>', styles['Normal']))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph('• Análisis de sangre completo', recommendation_style))
            story.append(Paragraph('• Tests de orina (completos)', recommendation_style))
            story.append(Paragraph('• Formulario médico estándar (IPSS, KDQOL o similar)', recommendation_style))

    doc.build(story)
    return pdf_filename
