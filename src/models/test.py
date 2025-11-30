"""
Módulo para evaluar modelos entrenados en el conjunto de test
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class CKDModelTester:
    """Clase para evaluar modelos en el conjunto de test"""

    def __init__(self, weights_dir='weights', results_dir='results', timestamp=None):
        """
        Inicializa el evaluador

        Args:
            weights_dir: Directorio con los modelos
            results_dir: Directorio para guardar resultados
            timestamp: Timestamp del entrenamiento (si None, usa el más reciente)
        """
        self.weights_dir = Path(weights_dir)
        self.results_dir = Path(results_dir)

        if timestamp is None:
            # Buscar el timestamp más reciente
            test_files = list(self.weights_dir.glob('test_data_*.csv'))
            if not test_files:
                raise ValueError("No se encontraron archivos de test")
            self.timestamp = sorted([f.stem.split('_')[-1] for f in test_files])[-1]
        else:
            self.timestamp = timestamp

        print(f"Usando timestamp: {self.timestamp}")

    def load_test_data(self):
        """Carga el conjunto de test guardado"""
        test_path = self.weights_dir / f'test_data_{self.timestamp}.csv'
        df = pd.read_csv(test_path)

        self.X_test = df.drop(columns=['Diagnosis'])
        self.y_test = df['Diagnosis']

        print(f"Test set cargado: {len(self.X_test)} muestras")
        return self.X_test, self.y_test

    def load_models(self):
        """Carga modelos y preprocessors"""
        # Cargar preprocessors
        prep_path = self.weights_dir / f'preprocessors_{self.timestamp}.pkl'
        with open(prep_path, 'rb') as f:
            prep = pickle.load(f)
            self.imputer = prep['imputer']
            self.scaler = prep['scaler']

        # Cargar thresholds
        thresh_path = self.weights_dir / f'thresholds_{self.timestamp}.json'
        with open(thresh_path, 'r') as f:
            self.thresholds = json.load(f)

        # Cargar modelos
        self.models = {}
        for model_name in ['logistic_regression', 'xgboost']:
            model_path = self.weights_dir / f'{model_name}_model_{self.timestamp}.pkl'
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)

        print(f"Modelos cargados: {list(self.models.keys())}")

    def preprocess_test_data(self):
        """Aplica el mismo preprocesamiento que en train"""
        self.X_test = pd.DataFrame(
            self.imputer.transform(self.X_test),
            columns=self.X_test.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )

    def evaluate_all_models(self):
        """Evalúa cada modelo con su preprocesamiento correcto"""
        results = {}

        for model_name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"EVALUANDO: {model_name.upper()}")
            print('=' * 80)

            # Seleccionar datos según el modelo
            if model_name == 'xgboost':
                X_test_model = self.X_test_nan_scaled  # Con NaN
            else:
                X_test_model = self.X_test_imputed_scaled  # Imputado

            y_pred_proba = model.predict_proba(X_test_model)[:, 1]

            # Evaluar con thresholds (mantener igual)
            for threshold_type, threshold in self.thresholds[model_name].items():
                y_pred = (y_pred_proba >= threshold).astype(int)

                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1_score': f1_score(self.y_test, y_pred),
                    'roc_auc': auc(*roc_curve(self.y_test, y_pred_proba)[:2]),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
                }

                if model_name not in results:
                    results[model_name] = {}
                results[model_name][threshold_type] = metrics

                print(f"\n--- Threshold {threshold_type}: {threshold:.4f} ---")
                print(f"Accuracy:  {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall:    {metrics['recall']:.4f}")
                print(f"F1-Score:  {metrics['f1_score']:.4f}")
                print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Guardar resultados
        results_path = self.results_dir / f'test_results_{self.timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results

def main():
    """Ejecuta la evaluación en test"""
    tester = CKDModelTester()
    tester.load_test_data()
    tester.load_models()
    tester.preprocess_test_data()
    results = tester.evaluate_all_models()

    print("\n✓ Evaluación completada")


if __name__ == '__main__':
    main()
