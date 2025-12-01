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
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api-patient-cdk-test', methods=['POST'])
def api_patient_cdk_test():
    # Obtener datos del formulario
    patient_id = environment.flask.request.form.get('patient_id')
    test_type = environment.flask.request.form.get('test_type')
    notes = environment.flask.request.form.get('notes', '')

    if not patient_id:
        return environment.flask.jsonify({"failed": True, "error": "Patient ID is required"}), 400

    # Obtener archivos PDF
    files = environment.flask.request.files.getlist('reports')

    if not files:
        return environment.flask.jsonify({"failed": True, "error": "No PDF files received"}), 400

    saved_files = []

    # Asegurar que existe el directorio tmp
    environment.paths.tmp.mkdir(parents=True, exist_ok=True)

    # Guardar cada archivo PDF en tmp
    blood_test_path = None
    for file in files:
        if file and file.filename:
            file_path = environment.paths.tmp.joinpath(file.filename)
            file.save(str(file_path))
            saved_files.append(file.filename)
            # Buscar específicamente Blood_Test_2025.pdf
            if file.filename == "Blood_Test_2025.pdf":
                blood_test_path = file_path

    # Procesar solo Blood_Test_2025.pdf
    patient_data = None
    if blood_test_path and blood_test_path.exists():
        try:
            patient_data = extractor.extract_patient_data_from_pdf(
                str(blood_test_path))

            # Enviar al modelo (puerto 8002)
            model_response = environment.requests.post(
                f"{environment.variables.model_api_address}/api-asses-risk",
                json={
                    'patient_id': patient_id,
                    'patient_data': patient_data
                },
                timeout=30
            )
            model_result = model_response.json() if model_response.status_code == 200 else None
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
        'processed_file': 'Blood_Test_2025.pdf' if blood_test_path else None,
        'data_extracted': patient_data,
        'model_prediction': model_result
    }

    response = environment.flask.make_response(
        environment.flask.jsonify(response_data))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


if __name__ == '__main__':
    app.run(
        host=environment.variables.server_ip,
        port=environment.variables.server_port,
        debug=(environment.log_level == environment.logging.DEBUG),
    )
    environment.unload_env()
