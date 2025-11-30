"""
Script para predecir CKD desde un archivo JSON usando XGBoost entrenado
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path


def _interpret_feature_value(feature_name, value):
    """Interpreta el valor de una característica con contexto clínico"""
    if value is None:
        return "Valor no disponible - No se puede evaluar"

    # Diccionario de interpretaciones clínicas
    interpretations = {
        'SerumCreatinine': lambda v: f"Normal ({v:.2f} mg/dL)" if 0.6 <= v <= 1.2 else f"Alterado ({v:.2f} mg/dL) - Requiere atención",
        'GFR': lambda v: f"Normal - Función renal óptima ({v:.1f} mL/min)" if v >= 90 else f"Reducido ({v:.1f} mL/min) - Posible deterioro renal",
        'BUNLevels': lambda v: f"Normal ({v:.1f} mg/dL)" if 7 <= v <= 20 else f"Alterado ({v:.1f} mg/dL)",
        'FastingBloodSugar': lambda v: f"Normal ({v:.0f} mg/dL)" if v < 100 else f"Elevado ({v:.0f} mg/dL) - Riesgo de diabetes",
        'HemoglobinLevels': lambda v: f"Normal ({v:.1f} g/dL)" if 12 <= v <= 17 else f"Bajo ({v:.1f} g/dL) - Posible anemia"
    }

    if feature_name in interpretations:
        return interpretations[feature_name](value)

    return f"Valor: {value}"


def _get_top_features_explanation(feature_names, importances, df_original, top_n=10):
    """Obtiene las top N características más importantes con explicaciones clínicas"""
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = []

    for idx in indices:
        feature_name = feature_names[idx]
        importance = float(importances[idx])
        raw_value = df_original[feature_name].iloc[0]

        # Convertir numpy types a Python natives o None
        if pd.isna(raw_value):
            value = None
        elif isinstance(raw_value, (np.integer, np.floating, np.number)):
            value = float(raw_value)
        else:
            value = str(raw_value)

        interpretation = _interpret_feature_value(feature_name, value)

        top_features.append({
            'feature': feature_name,
            'importance': importance,
            'value': value,
            'interpretation': interpretation
        })

    return top_features


def _interpret_probability(prob, threshold, available_features_count=0, total_features=10, patient_age=None):
    """
    Interpreta la probabilidad considerando datos disponibles y edad

    Args:
        prob: Probabilidad predicha
        threshold: Threshold usado
        available_features_count: Número de características disponibles
        total_features: Total de características
        patient_age: Edad del paciente (None si no está disponible)
    """
    margin = prob - threshold
    data_completeness = available_features_count / total_features

    # Determinar si necesita monitoreo por edad
    needs_monitoring = patient_age is not None and patient_age >= 50

    # Disclaimer por datos limitados
    if data_completeness < 0.5:
        disclaimer = " (ADVERTENCIA: Predicción con datos limitados - Se recomienda completar evaluación)"
    else:
        disclaimer = ""

    if prob < threshold:
        # Predicción: No CKD
        if prob < threshold * 0.3:
            # Riesgo muy bajo - Sin monitoreo
            return f"Riesgo muy bajo de CKD - Función renal normal. No se requiere monitoreo adicional{disclaimer}"
        elif prob < threshold * 0.6:
            # Riesgo bajo
            if needs_monitoring:
                return f"Riesgo bajo de CKD - Monitoreo preventivo anual recomendado por edad avanzada (50+){disclaimer}"
            else:
                return f"Riesgo bajo de CKD - Función renal saludable{disclaimer}"
        else:
            # Riesgo medio-bajo
            if needs_monitoring:
                return f"Riesgo medio-bajo de CKD - Monitoreo semestral recomendado por edad avanzada (50+){disclaimer}"
            else:
                return f"Riesgo medio-bajo de CKD - Mantener hábitos saludables{disclaimer}"
    else:
        # Predicción: CKD
        remaining_to_certain = 1.0 - threshold

        if prob > threshold + remaining_to_certain * 0.5:
            return f"Riesgo alto de CKD - Se recomienda evaluación nefrológica completa (análisis de orina y perfil renal extendido){disclaimer}"
        elif prob > threshold + remaining_to_certain * 0.2:
            return f"Riesgo moderado-alto de CKD - Se sugieren pruebas adicionales (panel metabólico completo y microalbuminuria){disclaimer}"
        else:
            return f"Riesgo moderado de CKD - Análisis de seguimiento recomendados{disclaimer}"


def predict_ckd_from_json(patient_data, weights_dir='weights', timestamp=20251130_000518):
    """
    Predice CKD desde un archivo JSON usando XGBoost

    Args:
        json_path: Ruta al archivo JSON con datos del paciente
        weights_dir: Directorio con los modelos
        timestamp: Timestamp del modelo (si None, usa el más reciente)

    Returns:
        dict con probabilidad y diagnóstico
    """
    weights_dir = Path(weights_dir)

    # Si no hay timestamp, buscar el más reciente
    if timestamp is None:
        model_files = list(weights_dir.glob('xgboost_model_*.pkl'))
        if not model_files:
            raise ValueError("No se encontraron modelos entrenados")
        timestamps = [f.stem.split('_')[-1] for f in model_files]
        timestamp = sorted(timestamps)[-1]

    print(f"Usando modelo con timestamp: {timestamp}")


    print(f"Datos del paciente cargados desde: patient_data")
    # 2. Convertir a DataFrame
    df = pd.DataFrame([patient_data])
    print("Datos del paciente convertidos a DataFrame")
    # 3. Guardar copia original con -1 para explicaciones
    df_original = df.copy()
    print("Copia original de datos guardada para explicaciones")
    # 4. Eliminar columnas no features
    cols_to_drop = ['PatientID', 'DoctorInCharge', 'Diagnosis']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df_original = df_original.drop(columns=[col for col in cols_to_drop if col in df_original.columns])

    # MOSTRAR VALORES DISPONIBLES AL INICIO
    print("\n" + "=" * 80)
    print("VALORES CLINICOS RECIBIDOS")
    print("=" * 80)
    available_values = {}
    for col in df.columns:
        val = df[col].iloc[0]
        if val != -1:
            available_values[col] = val
            print(f"  - {col}: {val}")

    if not available_values:
        print("  (No hay valores disponibles - Solo valores faltantes)")
    print(f"\nTotal de valores disponibles: {len(available_values)} de {len(df.columns)}")

    # 5. Convertir -1 a NaN (XGBoost los maneja nativamente)
    df = df.replace(-1, np.nan)
    print("Valores faltantes convertidos a NaN")
    # 6. Cargar preprocessors
    prep_path = weights_dir / f'preprocessors_20251130_000518.pkl'
    with open(prep_path, 'rb') as f:
        prep = pickle.load(f)
        scaler = prep['scaler']
        feature_names = prep['feature_names']
    print("Preprocesadores cargados")
    # 7. Asegurar orden correcto de features y añadir columnas faltantes
    df = df.reindex(columns=feature_names, fill_value=np.nan)
    df_original = df_original.reindex(columns=feature_names, fill_value=-1)

    print("Datos del paciente ordenados según features del modelo")
    # 8. Escalar (el scaler maneja NaN)
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_names
    )
    print("Datos del paciente escalados")
    # 9. Cargar modelo XGBoost
    model_path = weights_dir / f'xgboost_model_20251130_000518.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 10. Predecir (XGBoost maneja NaN nativamente)
    y_proba = model.predict_proba(df_scaled)[:, 1][0]

    # 11. Cargar threshold
    thresh_path = weights_dir / f'thresholds_20251130_000518.json'
    with open(thresh_path, 'r') as f:
        thresholds = json.load(f)
        #threshold = thresholds['xgboost']['probabilistic']
        threshold = thresholds['xgboost']['youden']
        threshold_type = 'youden'

    # 12. Hacer predicción binaria
    prediction = int(y_proba >= threshold)

    # 13. Calcular confianza ajustada según tipo de threshold
    if threshold_type == 'youden':
        distance = abs(y_proba - threshold)
        max_distance = max(threshold, 1.0 - threshold)
        confidence = float(min(distance / max_distance, 1.0))
    else:
        if y_proba >= threshold:
            distance = y_proba - threshold
            max_distance = 1.0 - threshold
        else:
            distance = threshold - y_proba
            max_distance = threshold
        confidence = float(min(distance / max_distance, 1.0))

    # 14. Obtener feature importance y valores del paciente
    feature_importance = model.feature_importances_
    top_features = _get_top_features_explanation(
        feature_names, feature_importance, df_original, top_n=10
    )

    # Contar valores disponibles (no None)
    available_count = sum(1 for f in top_features if f['value'] is not None)

    # Extraer edad del paciente si está disponible
    patient_age = patient_data.get('Age')
    if patient_age == -1:
        patient_age = None

    # 15. Preparar resultado
    distance = abs(float(y_proba - threshold))
    result = {
        'patient_id': patient_data.get('PatientID', 'Unknown'),
        'probability': float(y_proba),
        'diagnosis': 'CKD' if prediction == 1 else 'No CKD',
        'threshold_used': float(threshold),
        'threshold_type': threshold_type,
        'confidence': confidence,
        'distance_to_threshold': distance,
        'interpretation': _interpret_probability(
            y_proba, threshold, available_count, len(top_features), patient_age
        ),
        'top_features': top_features,
        'data_completeness': available_count / len(top_features)
    }

    return result


def main():
    """Ejecuta la predicción"""
    JSON_PATH = '../models/patient_data_extracted.json'
    WEIGHTS_DIR = 'weights'

    result = predict_ckd_from_json(JSON_PATH, WEIGHTS_DIR)

    # Mostrar resultados
    print("\n" + "=" * 80)
    print("RESULTADO DE PREDICCIÓN CKD")
    print("=" * 80)
    print(f"Paciente ID: {result['patient_id']}")
    print(f"\nProbabilidad de CKD: {result['probability']:.4f}")
    print(f"Diagnóstico: {result['diagnosis']}")
    print(f"Confianza: {result['confidence']:.2%}")
    print(f"Distancia al threshold: {result['distance_to_threshold']:.4f}")

    print(f"\n{result['interpretation']}")
    print(f"\n(Threshold usado: {result['threshold_used']:.4f} - Tipo: {result['threshold_type'].upper()})")

    #solo printear si tiene las top 10 características el paciente evaluado
    if len(result['top_features']) <10:
        print("\n" + "=" * 80)
        print("TOP 10 CARACTERÍSTICAS MÁS RELEVANTES PARA EL DIAGNÓSTICO")
        print("=" * 80)
        for i, feature in enumerate(result['top_features'], 1):
            importance_bar = "█" * int(feature['importance'] * 40)
            print(f"\n{i}. {feature['feature']}")
            print(f"   Importancia: {importance_bar} {feature['importance']:.3f}")
            print(f"   {feature['interpretation']}")

    # VALORES DISPONIBLES DEL PACIENTE
    #print("\n" + "=" * 80)
    #print("VALORES CLÍNICOS DISPONIBLES DEL PACIENTE")
    #print("=" * 80)

    available_features = [(f['feature'], f['value'], f['interpretation'])
                         for f in result['top_features']
                         if f['value'] is not None]

    if available_features:
        #print(f"\n{len(available_features)} valores disponibles de las top 10 características:")
        #for feature_name, value, interpretation in available_features:
            #print(f"\n• {feature_name}")
            #print(f"  Valor: {value}")
            #print(f"  {interpretation}")

        # Advertencia si hay muy pocos datos
        completeness = len(available_features) / len(result['top_features'])
        if completeness < 0.5:
            print(f"\nATENCION: Solo el {completeness:.0%} de las características principales están disponibles")
            print("   La predicción puede no ser precisa. Se recomienda:")
            print("   • Análisis de orina completo")
            print("   • Perfil lipídico")
            print("   • Presión arterial")
            print("   • Historial clínico completo")
    else:
        print("\nNo hay valores disponibles en las top 10 características")
        print("   Recomendación: Completar evaluación clínica para mejorar precisión")

    print("\n" + "=" * 80)

    # Guardar resultado
    output_path = Path('../models/prediction_result.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nResultado guardado en: {output_path}")

    return result['probability']


if __name__ == '__main__':
    probability = main()
