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
        environment.flask.jsonify({'test': 'on'}))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api-asses-risk', methods=['POST'])
def api_asses_risk():
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

        # Extraer patient_data
        if 'patient_data' in data:
            patient_data = data['patient_data']
        else:
            patient_data = data

        # Ejecutar el modelo
        resultado = algorithm.predict_ckd_from_json(
            patient_data,
            weights_dir='weights',
        )

        # Enviar resultado a la API de respuesta usando variable de entorno
        response_api = f'{environment.variables.response_api_address}'
        + '/api-send-response'
        try:
            response = requests.post(
                response_api,
                json={
                    'patient_id': data.get('patient_id'),
                    'result': resultado
                },
                timeout=5,
            )

        except requests.exceptions.RequestException:
            pass

        # Devolver respuesta al cliente original
        response = environment.flask.make_response(
            environment.flask.jsonify(
                {
                    'status': 'success',
                    'result': resultado,
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
