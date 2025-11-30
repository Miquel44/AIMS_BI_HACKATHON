# -*- coding: utf-8 -*- noqa

import environment
import requests
import algorithm
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


@app.route('/api-asses-risk', methods=['POST'])
def api_asses_risk():
    try:
        # Obtener el JSON del request
        data = environment.flask.request.get_json()

        if not data:
            return environment.flask.jsonify({
                "error": "No se recibió ningún JSON"
            }), 400

        # Extraer patient_data
        if 'patient_data' in data:
            patient_data = data['patient_data']
        else:
            patient_data = data

        # Ejecutar el modelo
        resultado = algorithm.predict_ckd_from_json(
            patient_data,
            weights_dir='weights'
        )
        print("Processed result:", resultado)

        # Enviar resultado a la API de respuesta usando variable de entorno
        response_api = f"{environment.variables.response_api_address}/api-send-response"
        try:
            response = requests.post(
                response_api,
                json={
                    "patient_id": data.get('patient_id'),
                    "result": resultado
                },
                timeout=5
            )
            print(f"Resultado enviado a {response_api}")
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            print(f"Response Body: {response.text}")

            if response.status_code != 200:
                print(f"ERROR: La API respondió con código {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error al enviar resultado: {e}")
            import traceback
            traceback.print_exc()

        # Devolver respuesta al cliente original
        return environment.flask.jsonify({
            "status": "success",
            "result": resultado
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return environment.flask.jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(
        host=environment.variables.server_ip,
        port=environment.variables.server_port,
        debug=(environment.log_level == environment.logging.DEBUG),
    )
    environment.unload_env()
