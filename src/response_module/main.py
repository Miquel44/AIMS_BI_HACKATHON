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

        # Generate PDF using dedicated helper
        from .report import generate_report

        pdf_filename = generate_report(patient_id, result)

        medical_api = environment.variables.medical_api_address
        upload_url = f'{medical_api}/api-save-patient-report'

        with open(pdf_filename, 'rb') as pdf_file:
            files = {'report': (environment.os.path.basename(
                pdf_filename), pdf_file, 'application/pdf')}
            form_data = {
                'patient_id': patient_id,
                'file_name': environment.os.path.basename(pdf_filename)
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
