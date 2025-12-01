# -*- coding: utf-8 -*- noqa

import environment
import extractor

if __name__ == '__main__':
    environment.load_env()

app = environment.flask.Flask(__name__)


def main():
    pass


@app.route('/')
def origin():
    response = environment.flask.make_response(
        environment.flask.jsonify({"test": "on"}))
    response.status_code = 200
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api-patient-cdk-test', methods=['POST'])
def api_patient_cdk_test():
    # Obtener datos del formulario
    patient_id = environment.flask.request.form.get('patient_id')
    notes = environment.flask.request.form.get('notes', '')

    if not patient_id:
        response = environment.flask.make_response(environment.flask.jsonify(
            {"failed": True, "error": "Patient ID is required"},
        ))
        response.status_code = 400
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    # Obtener archivos PDF
    files = environment.flask.request.files.getlist('reports')

    if not files:
        response = environment.flask.make_response(environment.flask.jsonify(
            {"failed": True, "error": "No PDF files received"},
        ))
        response.status_code = 400
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    saved_files = []

    # Guardar cada archivo PDF en tmp
    blood_test_path = None
    for file in files:
        if file and file.filename:
            file_path = environment.paths.tmp.joinpath(file.filename)
            file.save(str(file_path))
            saved_files.append(file.filename)
            # Buscar específicamente Blood Test
            if file.filename.find('Blood') != -1:
                blood_test_path = file_path

    # Procesar solo Blood_Test_2025.pdf
    patient_data = None
    if blood_test_path and blood_test_path.exists():
        try:
            patient_data = extractor.extract_patient_data_from_pdf(
                str(blood_test_path),
            )

            # Enviar al modelo
            model_response = environment.requests.post(
                f'{environment.variables.model_api_address}/api-asses-risk',
                json={
                    'patient_id': patient_id,
                    'patient_data': patient_data,
                },
                timeout=30,
            )
            if model_response.status_code == 200:
                model_result = model_response.json()
            else:
                None
        except Exception:
            model_result = None
        finally:
            for file in files:
                if file and file.filename:
                    file_path = environment.paths.tmp.joinpath(file.filename)
                    if file_path.exists():
                        file_path.unlink()

    else:
        model_result = None

    # Respuesta exitosa
    response_data = {
        'failed': False,
        'message': 'CKD test completed',
        'patient_id': patient_id,
        'files_saved': saved_files,
        'processed_file': blood_test_path,
        'data_extracted': patient_data,
        'model_prediction': model_result,
    }

    response = environment.flask.make_response(
        environment.flask.jsonify(response_data),
    )
    response.status_code = 200
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
