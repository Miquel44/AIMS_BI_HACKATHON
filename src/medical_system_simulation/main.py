# -*- coding: utf-8 -*- noqa

import environment

if __name__ == '__main__':
    environment.load_env()

app = environment.flask.Flask(__name__)


def main():
    pass


@app.route('/')
def index():
    patients_path = environment.paths.data.joinpath('patients')

    patient_list = []

    if patients_path.exists():
        for item in patients_path.iterdir():
            if item.is_dir():
                data_file = item.joinpath("data.json")
                if data_file.exists():
                    try:
                        with open(data_file, "r", encoding="utf-8") as f:
                            data = environment.json.load(f)
                            patient_list.append(data)
                    except Exception as e:
                        print("Error loading", data_file, e)

    response = environment.flask.make_response(
        environment.flask.render_template(
            'index.html',
            patients=patient_list
        )
    )
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response


@app.route('/patient-record')
def patient_record():
    arguments = environment.flask.request.args.to_dict()

    if arguments.get('patient_id'):
        patient_path = environment.paths.data.joinpath(
            'patients',
            str(arguments['patient_id'])
        )

        data_file = patient_path.joinpath("data.json")

        if patient_path.exists() and data_file.exists():

            # Load the JSON file
            with open(data_file, "r", encoding="utf-8") as f:
                patient_data = environment.json.load(f)

            # Pass patient_data into template
            response = environment.flask.make_response(
                environment.flask.render_template(
                    'patient_record.html',
                    params=arguments,
                    patient=patient_data,     # <-- add this
                ),
            )
            response.headers['Content-Type'] = 'text/html; charset=utf-8'
            return response

    return "Patient not found", 404


@app.route('/api-get-patient-report')
def api_get_patient_report():
    arguments = environment.flask.request.args.to_dict()

    data = {
        'failed': False,
    }

    if arguments.get('patient_id', False) and arguments.get('report'):
        patient_report_path = environment.paths.data.joinpath(
            'patients',
            str(arguments['patient_id']),
            'reports',
            arguments['report'],
        )

        if patient_report_path.exists():
            return environment.flask.send_file(
                patient_report_path,
                as_attachment=False,
            )

        else:
            data['failed'] = True

    else:
        data['failed'] = True

    response = environment.flask.make_response(environment.flask.jsonify(data))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-get-patient-reports')
def api_get_patient_reports():
    arguments = environment.flask.request.args.to_dict()

    data = {
        'failed': False,
        'reports': [],
    }

    if arguments.get('patient_id', False):
        patient_reports_path = environment.paths.data.joinpath(
            'patients',
            str(arguments['patient_id']),
            'reports',
        )

        if patient_reports_path.exists():
            for path in patient_reports_path.iterdir():
                if path.is_file():
                    name = path.parts[-1]
                    data['reports'].append(
                        {
                            'name': name,
                            'url': f'http://localhost:8000/api-get-patient-report?patient_id={arguments["patient_id"]}&report={name}'},
                    )

        else:
            data['failed'] = True

    else:
        data['failed'] = True

    response = environment.flask.make_response(environment.flask.jsonify(data))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-request-patient-test', methods=['POST'])
def api_request_patient_test():
    data_received = environment.flask.request.get_json()

    if data_received is None:
        return environment.flask.jsonify({"failed": True, "error": "No JSON received"}), 400

    # Log para debug
    print("Received test request:")
    print(f"  Patient ID: {data_received.get('patient_id')}")
    print(f"  Test Type: {data_received.get('test_type')}")
    print(f"  Notes: {data_received.get('notes')}")
    if 'reports' in data_received:
        print(f"  Reports: {len(data_received.get('reports', []))} files")

    # Respuesta exitosa
    response_data = {
        'failed': False,
        'message': 'Test request received successfully',
        'test_type': data_received.get('test_type'),
        'patient_id': data_received.get('patient_id'),
    }

    # Si es CKD Risk Assessment, enviar al módulo extractor
    if data_received.get('test_type') == 'CKD Risk Assesment':
        print("  -> Forwarding to extractor module (port 8001)...")
        try:
            # Preparar datos del formulario
            form_data = {
                'patient_id': data_received.get('patient_id'),
                'test_type': data_received.get('test_type'),
                'notes': data_received.get('notes', '')
            }

            # Preparar archivos
            files = []
            for report in data_received.get('reports', []):
                # Obtener el archivo desde el path local
                patient_id = data_received.get('patient_id')
                report_name = report.get('name')

                if patient_id and report_name:
                    report_path = environment.paths.data.joinpath(
                        'patients',
                        str(patient_id),
                        'reports',
                        report_name
                    )

                    if report_path.exists():
                        # Leer el archivo y agregarlo a la lista
                        with open(report_path, 'rb') as f:
                            file_content = f.read()
                            files.append(('reports', (report_name, file_content, 'application/pdf')))

            # Enviar como multipart/form-data
            extractor_response = environment.requests.post(
                'http://localhost:8001/api-patient-cdk-test',
                data=form_data,
                files=files,
                timeout=30
            )
            print(f"  -> Extractor response: {extractor_response.status_code}")

        except Exception as e:
            print(f"  -> Error sending to extractor: {e}")

    response = environment.flask.make_response(
        environment.flask.jsonify(response_data))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-save-patient-report', methods=['POST'])
def api_save_patient_report():
    # Expecting multipart/form-data:
    #   - patient_id (text)
    #   - file_name (text)
    #   - report (file)

    patient_id = environment.flask.request.form.get("patient_id")
    file_name = environment.flask.request.form.get("file_name")
    uploaded_file = environment.flask.request.files.get("report")

    # Validate fields
    if not patient_id or not file_name or uploaded_file is None:
        return environment.flask.jsonify({
            "failed": True,
            "error": "Missing patient_id, file_name, or report file"
        }), 400

    # Build the directory path
    reports_path = environment.paths.data.joinpath(
        "patients",
        str(patient_id),
        "reports"
    )

    # Ensure directory exists
    try:
        reports_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return environment.flask.jsonify({
            "failed": True,
            "error": f"Cannot create reports directory: {e}"
        }), 500

    # Save the file
    try:
        save_path = reports_path.joinpath(file_name)
        uploaded_file.save(str(save_path))

    except Exception as e:
        return environment.flask.jsonify({
            "failed": True,
            "error": f"Error saving file: {e}"
        }), 500

    # Success response
    return environment.flask.jsonify({
        "failed": False,
        "message": "Report saved successfully",
        "file": file_name,
        "patient_id": patient_id
    })


if __name__ == '__main__':
    app.run(
        host=environment.variables.server_ip,
        port=environment.variables.server_port,
        debug=(environment.log_level == environment.logging.DEBUG),
    )
    environment.unload_env()
