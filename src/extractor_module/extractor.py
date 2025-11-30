import environment
from google import genai
from google.genai import types
from google.genai.errors import APIError
import textwrap

# Ruta del archivo que contiene la API key de Gemini (siempre absoluta)
GEMINI_API_KEY_PATH = environment.ENVIRONMENT_PATH.joinpath('gemini_api.key')


def load_gemini_api_key(path: str) -> str:
    """Lee la API key desde un archivo de texto y la guarda en la variable de entorno."""
    path = environment.os.path.normpath(path)
    if not environment.os.path.isabs(path):
        print(
            f"Se requiere una ruta absoluta para la API key. Ruta dada: '{path}'")
        exit()

    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if not key:
            raise ValueError("El archivo de la API key está vacío")
        environment.os.environ["GEMINI_API_KEY"] = key
        return key
    except Exception as e:
        print(f"Error al leer la API key desde '{path}': {e}")
        exit()


# Plantilla base del JSON (todos los campos en -1)
PATIENT_DATA_TEMPLATE = {
    "PatientID": -1,
    "Age": -1,
    "Gender": -1,
    "Ethnicity": -1,
    "SocioeconomicStatus": -1,
    "EducationLevel": -1,
    "BMI": -1,
    "Smoking": -1,
    "AlcoholConsumption": -1,
    "PhysicalActivity": -1,
    "DietQuality": -1,
    "SleepQuality": -1,
    "FamilyHistoryKidneyDisease": -1,
    "FamilyHistoryHypertension": -1,
    "FamilyHistoryDiabetes": -1,
    "PreviousAcuteKidneyInjury": -1,
    "UrinaryTractInfections": -1,
    "SystolicBP": -1,
    "DiastolicBP": -1,
    "FastingBloodSugar": -1,
    "HbA1c": -1,
    "SerumCreatinine": -1,
    "BUNLevels": -1,
    "GFR": -1,
    "ProteinInUrine": -1,
    "ACR": -1,
    "SerumElectrolytesSodium": -1,
    "SerumElectrolytesPotassium": -1,
    "SerumElectrolytesCalcium": -1,
    "SerumElectrolytesPhosphorus": -1,
    "HemoglobinLevels": -1,
    "CholesterolTotal": -1,
    "CholesterolLDL": -1,
    "CholesterolHDL": -1,
    "CholesterolTriglycerides": -1,
    "ACEInhibitors": -1,
    "Diuretics": -1,
    "NSAIDsUse": -1,
    "Statins": -1,
    "AntidiabeticMedications": -1,
    "Edema": -1,
    "FatigueLevels": -1,
    "NauseaVomiting": -1,
    "MuscleCramps": -1,
    "Itching": -1,
    "QualityOfLifeScore": -1,
    "HeavyMetalsExposure": -1,
    "OccupationalExposureChemicals": -1,
    "WaterQuality": -1,
    "MedicalCheckupsFrequency": -1,
    "MedicationAdherence": -1,
    "HealthLiteracy": -1,
    "Diagnosis": -1,
    "DoctorInCharge": -1
}


def extract_patient_data_from_pdf(pdf_path: str) -> dict:
    """Extrae datos de paciente desde un PDF usando Gemini."""

    # Inicializar cliente
    load_gemini_api_key(GEMINI_API_KEY_PATH)
    client = genai.Client()

    # Verificar que el PDF existe
    if not environment.os.path.exists(pdf_path):
        print(f"Error: PDF no encontrado en {pdf_path}")
        return PATIENT_DATA_TEMPLATE.copy()

    # Subir el archivo a Gemini
    print(f"Subiendo PDF: {pdf_path}")

    uploaded_file = client.files.upload(file=pdf_path)

    # Crear el prompt
    prompt = textwrap.dedent(f"""
        Eres un experto en extracción de datos médicos. Analiza el documento adjunto (historial clínico, informe médico o resultados de laboratorio).

        Extrae ÚNICAMENTE los siguientes datos del paciente que encuentres en el documento:

        {environment.json.dumps(list(PATIENT_DATA_TEMPLATE.keys()), indent=2)}

        INSTRUCCIONES IMPORTANTES:
        - Si un dato NO aparece en el documento, déjalo como -1
        - Devuelve ÚNICAMENTE un objeto JSON válido
        - Para valores booleanos (Smoking, AlcoholConsumption, etc.), usa: 1 para Sí, 0 para No, -1 para desconocido
        - Para valores numéricos, extrae el número exacto
        - Para texto (Gender, Ethnicity, etc.), extrae el valor tal cual aparece
        - NO agregues explicaciones, solo el JSON

        Ejemplo de respuesta esperada:
        {{
          "PatientID": "12345",
          "Age": 45,
          "Gender": "Male",
          "SystolicBP": 140,
          "DiastolicBP": 90,
          "SerumCreatinine": 1.5,
          "GFR": 60,
          ...resto de campos en -1 si no se encuentran...
          Es importante no sobreescribir ningún campo que no se encuentre en el documento.
        }}
    """)

    # Hacer la consulta a Gemini
    print("Consultando a Gemini...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type
                ),
                prompt
            ]
        )

        # Extraer el JSON de la respuesta
        response_text = response.text.strip()

        # Limpiar markdown code blocks si existen
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        # Parsear JSON
        extracted_data = environment.json.loads(response_text.strip())

        # Combinar con plantilla (para asegurar todos los campos)
        result = PATIENT_DATA_TEMPLATE.copy()
        result.update(extracted_data)

        print("Datos extraídos exitosamente")
        return result

    except environment.json.JSONDecodeError as e:
        print(f"Error parseando JSON: {e}")
        print(f"Respuesta recibida: {response.text[:500]}")
        return PATIENT_DATA_TEMPLATE.copy()

    except APIError as e:
        print(f"Error de API de Gemini: {e}")
        return PATIENT_DATA_TEMPLATE.copy()

    finally:
        # Limpiar archivo subido
        client.files.delete(name=uploaded_file.name)


# Ejemplo de uso
if __name__ == '__main__':
    PDF_PATH = r"C:\Users\mique\OneDrive\Escritorio\Hackathon\Equipo-equipo-aims-22\src\extractor_module\pdftmp\Blood_Test_2025.pdf"

    patient_data = extract_patient_data_from_pdf(PDF_PATH)

    print("\n" + "=" * 50)
    print("DATOS DEL PACIENTE EXTRAÍDOS:")
    print("=" * 50)
    print(environment.json.dumps(patient_data, indent=2, ensure_ascii=False))

    # Guardar en archivo JSON
    output_path = "pdftmp/patient_data_extracted.json"
    with open(output_path, "w", encoding="utf-8") as f:
        environment.json.dump(patient_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Datos guardados en: {output_path}")
