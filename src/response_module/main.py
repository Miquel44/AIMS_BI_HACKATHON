# -*- coding: utf-8 -*- noqa

import environment

if __name__ == '__main__':
    environment.load_env()

app = environment.flask.Flask(__name__)


def main():
    pass


@app.route('/')
def origin():
    response = environment.flask.make_response(
        environment.flask.jsonify({'test': 'on'}))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api-send-response', methods=['POST'])
def api_send_response():
    try:
        # Obtener el JSON del request
        data = environment.flask.request.get_json()

        if not data:
            response = environment.flask.make_response(
                environment.flask.jsonify(
                    {
                        'error': 'No JSON recived',
                    },
                ),
            )
            response.status_code = 400
            response.headers['Content-Type'] = 'application/json; charset=utf-8'
            return response

        # Extraer datos
        patient_id = data.get('patient_id', 'N/A')
        result = data.get('result', {})

        # Generar PDF
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from datetime import datetime
        import os

        # Crear directorio si no existe
        os.makedirs('reports', exist_ok=True)

        # Nombre del archivo PDF
        timestamp = datetime.now().strftime(environment.time_format)
        pdf_filename = f'reports/Report CKD {patient_id} {timestamp}.pdf'

        # Crear documento
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Estilo personalizado para título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=1  # Centrado
        )

        # Estilo para subtítulos
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12
        )

        # Título del reporte
        story.append(Paragraph('REPORTE DE EVALUACIÓN CKD', title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Información del paciente
        story.append(
            Paragraph(f'<b>Paciente ID:</b> {patient_id}', styles['Normal']))
        story.append(Paragraph(
            f'<b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}', styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Resultados principales
        story.append(Paragraph('RESULTADO DE PREDICCIÓN', subtitle_style))

        # Tabla con resultados principales
        prob = result.get('probability', 0)
        diagnosis = result.get('diagnosis', 'N/A')
        confidence = result.get('confidence', 0)

        # Color basado en diagnóstico
        if diagnosis == 'CKD':
            diag_color = colors.HexColor('#E74C3C')
        else:
            diag_color = colors.HexColor('#27AE60')

        main_data = [
            ['Métrica', 'Valor'],
            ['Probabilidad de CKD', f'{prob:.4f}'],
            ['Diagnóstico', diagnosis],
            ['Confianza', f'{confidence:.2%}'],
            ['Distancia al threshold', f'{
                result.get('distance_to_threshold', 0):.4f}'],
            ['Threshold usado',
             f'{result.get('threshold_used', 0):.4f} ({result.get('threshold_type', 'N/A').upper()})']
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

        # Interpretación
        interpretation = result.get('interpretation', '')
        if interpretation:
            story.append(Paragraph('INTERPRETACIÓN', subtitle_style))
            story.append(Paragraph(interpretation, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))

        # Top características con valores disponibles
        top_features = result.get('top_features', [])
        available_features = [f for f in top_features if f.get(
            'value') is not None and f.get('value') != -1]

        if available_features:
            story.append(
                Paragraph(f'CARACTERÍSTICAS MÁS RELEVANTES (Top 10)', subtitle_style))

            features_data = [['#', 'Característica', 'Valor', 'Importancia']]

            for i, feature in enumerate(available_features[:10], 1):
                importance = feature.get('importance', 0)
                importance_bar = '█' * int(importance * 20)

                value = feature.get('value', 'N/A')
                if isinstance(value, (int, float)):
                    value_str = f'{value:.4f}' if isinstance(
                        value, float) else str(value)
                else:
                    value_str = str(value)

                features_data.append([
                    str(i),
                    feature.get('feature', 'N/A'),
                    value_str,
                    f'{importance_bar} {importance:.3f}'
                ])

            features_table = Table(features_data, colWidths=[
                                   0.5 * inch, 2.5 * inch, 1.5 * inch, 2 * inch])
            features_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.lightgrey]),
            ]))

            story.append(features_table)
            story.append(Spacer(1, 0.3 * inch))

            # TABLA CON TODAS LAS CARACTERÍSTICAS (siempre mostrar si hay datos)
            story.append(Paragraph(f'TODAS LAS CARACTERÍSTICAS DISPONIBLES ({len(available_features)} total)',
                                   subtitle_style))

            all_features_data = [['#', 'Característica', 'Valor']]

            for i, feature in enumerate(available_features, 1):
                value = feature.get('value', 'N/A')
                if isinstance(value, (int, float)):
                    value_str = f'{value:.4f}' if isinstance(
                        value, float) else str(value)
                else:
                    value_str = str(value)

                all_features_data.append([
                    str(i),
                    feature.get('feature', 'N/A'),
                    value_str
                ])

            all_features_table = Table(all_features_data, colWidths=[
                                       0.5 * inch, 3.5 * inch, 2.5 * inch])
            all_features_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16A085')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.white, colors.lightgrey]),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))

            story.append(all_features_table)
            story.append(Spacer(1, 0.2 * inch))

            # Advertencia de completitud
            completeness = len(available_features) / \
                len(top_features) if top_features else 0
            if completeness < 0.5:
                warning_style = ParagraphStyle(
                    'Warning',
                    parent=styles['Normal'],
                    textColor=colors.HexColor('#E67E22'),
                    fontSize=10,
                    leading=14
                )

                total_possible = len(top_features)
                available_count = len(available_features)

                story.append(Paragraph(
                    f'<b>ATENCIÓN:</b> Solo {available_count}/{
                        total_possible} características principales están disponibles. '
                    f'El modelo puede tener en cuenta hasta {
                        total_possible} características para una evaluación más precisa.',
                    warning_style
                ))
                story.append(Spacer(1, 0.1 * inch))

                recommendation_style = ParagraphStyle(
                    'Recommendation',
                    parent=styles['Normal'],
                    fontSize=10,
                    leading=14,
                    leftIndent=20
                )

                story.append(
                    Paragraph('<b>Para un análisis más preciso siempre recomendamos completar:</b>', styles['Normal']))
                story.append(Spacer(1, 0.05 * inch))
                story.append(
                    Paragraph('• Análisis de sangre completo', recommendation_style))
                story.append(
                    Paragraph('• Tests de orina (completos)', recommendation_style))
                story.append(Paragraph(
                    '• Formulario médico estándar (IPSS, KDQOL o similar)', recommendation_style))
        # Construir PDF
        doc.build(story)

        medical_api = environment.variables.medical_api_address
        upload_url = f'{medical_api}/api-save-patient-report'

        with open(pdf_filename, 'rb') as pdf_file:
            files = {'report': (os.path.basename(
                pdf_filename), pdf_file, 'application/pdf')}
            form_data = {
                'patient_id': patient_id,
                'file_name': os.path.basename(pdf_filename)
            }

            response = environment.requests.post(
                upload_url, files=files, data=form_data)

        # Devolver confirmación
        response = environment.flask.make_response(
            environment.flask.jsonify(
                {
                    'status': 'success',
                    'message': 'Datos recibidos y PDF generado correctamente',
                    'pdf_path': pdf_filename,
                },
            ),
        )
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    except Exception as error:
        response = environment.flask.make_response(
            environment.flask.jsonify(
                {
                    'error': str(error),
                },
            ),
        )
        response.status_code = 500
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response


if __name__ == '__main__':
    try:
        app.run(
            host=environment.variables.server_ip,
            port=environment.variables.server_port,
            debug=(environment.log_level == environment.logging.DEBUG),
        )
    finally:
        environment.unload_env()
