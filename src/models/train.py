"""
Módulo de entrenamiento para modelos de predicción de Enfermedad Renal Crónica (CKD)
Incluye XGBoost y Regresión Logística con manejo de valores faltantes
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CKDModelTrainer:
    """Clase para entrenar modelos de predicción de CKD"""

    def __init__(self, data_path, results_dir='results', weights_dir='weights'):
        """
        Inicializa el entrenador de modelos

        Args:
            data_path: Ruta al archivo CSV con los datos
            results_dir: Directorio para guardar resultados
            weights_dir: Directorio para guardar pesos del modelo
        """
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.weights_dir = Path(weights_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)

        # Crear timestamp para esta ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Inicializar atributos
        self.df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None
        self.imputer = None
        self.scaler = None

        # Modelos
        self.models = {}
        self.thresholds = {}
        self.metrics = {}

    def load_and_prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split en train/val/test
        """
        print("=" * 80)
        print("CARGANDO Y PREPARANDO DATOS")
        print("=" * 80)

        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

        # Eliminar columnas no necesarias
        cols_to_drop = ['PatientID', 'DoctorInCharge']
        X = self.df.drop(columns=cols_to_drop + ['Diagnosis'])
        y = self.df['Diagnosis']

        self.feature_names = X.columns.tolist()

        # Split train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Guardar índices de test
        test_indices = X_test.index.tolist()
        np.save(self.weights_dir / f'test_indices_{self.timestamp}.npy', test_indices)

        # Split train / val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )

        print(f"\n✓ Datos divididos:")
        print(f"  - Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"  - Validación: {self.X_val.shape[0]} muestras")
        print(f"  - Test: {X_test.shape[0]} muestras")

        # Guardar test set completo
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv(self.weights_dir / f'test_data_{self.timestamp}.csv', index=False)


        # Configurar imputer para valores faltantes (media para numéricas)
        self.imputer = SimpleImputer(strategy='mean')
        self.X_train = pd.DataFrame(
            self.imputer.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_val = pd.DataFrame(
            self.imputer.transform(self.X_val),
            columns=self.feature_names
        )

        # Estandarizar features
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_names
        )

        print("✓ Datos imputados y estandarizados")

    def calculate_youden_threshold(self, y_true, y_pred_proba):
        """
        Calcula el threshold óptimo usando el índice de Youden

        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas

        Returns:
            threshold óptimo, índice de Youden
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold, youden_index[optimal_idx]

    def calculate_probabilistic_threshold(self, y_true, y_pred_proba):
        """
        Calcula threshold probabilístico que maximiza F1-score

        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas

        Returns:
            threshold óptimo, F1-score máximo
        """
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold, f1_scores[optimal_idx]

    def train_logistic_regression(self, max_iter=1000):
        """
        Entrena modelo de Regresión Logística

        Args:
            max_iter: Número máximo de iteraciones
        """
        print("\n" + "=" * 80)
        print("ENTRENANDO REGRESIÓN LOGÍSTICA")
        print("=" * 80)

        # Entrenar modelo
        lr_model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        lr_model.fit(self.X_train, self.y_train)

        # Validación cruzada
        cv_scores = cross_val_score(lr_model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
        print(f"\n✓ Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Predicciones en validación
        y_pred_proba = lr_model.predict_proba(self.X_val)[:, 1]

        # Calcular thresholds
        youden_threshold, youden_idx = self.calculate_youden_threshold(self.y_val, y_pred_proba)
        prob_threshold, max_f1 = self.calculate_probabilistic_threshold(self.y_val, y_pred_proba)

        print(f"\n✓ Threshold Youden: {youden_threshold:.4f} (Índice: {youden_idx:.4f})")
        print(f"✓ Threshold Probabilístico (max F1): {prob_threshold:.4f} (F1: {max_f1:.4f})")

        # Guardar modelo y thresholds
        self.models['logistic_regression'] = lr_model
        self.thresholds['logistic_regression'] = {
            'youden': float(youden_threshold),
            'probabilistic': float(prob_threshold)
        }

        # Evaluar con ambos thresholds
        self._evaluate_model('logistic_regression', y_pred_proba)

    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Entrena modelo XGBoost con manejo nativo de valores faltantes
        """
        print("\n" + "=" * 80)
        print("ENTRENANDO XGBOOST")
        print("=" * 80)

        # Convertir -1 a NaN en datos de entrenamiento y validación
        X_train_nan = self.X_train.replace(-1, np.nan)
        X_val_nan = self.X_val.replace(-1, np.nan)

        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            missing=np.nan  # ← CLAVE: XGBoost maneja NaN nativamente
        )

        xgb_model.fit(
            X_train_nan, self.y_train,
            eval_set=[(X_val_nan, self.y_val)],
            verbose=False
        )

        # Validación cruzada
        cv_scores = cross_val_score(xgb_model, X_train_nan, self.y_train, cv=5, scoring='roc_auc')
        print(f"\n✓ Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Predicciones con NaN
        y_pred_proba = xgb_model.predict_proba(X_val_nan)[:, 1]

        # Calcular thresholds (mantener como está)
        youden_threshold, youden_idx = self.calculate_youden_threshold(self.y_val, y_pred_proba)
        prob_threshold, max_f1 = self.calculate_probabilistic_threshold(self.y_val, y_pred_proba)

        print(f"\n✓ Threshold Youden: {youden_threshold:.4f} (Índice: {youden_idx:.4f})")
        print(f"✓ Threshold Probabilístico (max F1): {prob_threshold:.4f} (F1: {max_f1:.4f})")

        self.models['xgboost'] = xgb_model
        self.thresholds['xgboost'] = {
            'youden': float(youden_threshold),
            'probabilistic': float(prob_threshold)
        }

        self._evaluate_model('xgboost', y_pred_proba)

    def _evaluate_model(self, model_name, y_pred_proba):
        """
        Evalúa un modelo con diferentes thresholds

        Args:
            model_name: Nombre del modelo
            y_pred_proba: Probabilidades predichas
        """
        print(f"\n{'=' * 80}")
        print(f"EVALUACIÓN: {model_name.upper()}")
        print("=" * 80)

        thresholds = self.thresholds[model_name]
        self.metrics[model_name] = {}

        for threshold_type, threshold_value in thresholds.items():
            print(f"\n--- Threshold {threshold_type.upper()}: {threshold_value:.4f} ---")

            y_pred = (y_pred_proba >= threshold_value).astype(int)

            # Calcular métricas
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred)
            recall = recall_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            roc_auc = auc(*roc_curve(self.y_val, y_pred_proba)[:2])

            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")

            # Guardar métricas
            self.metrics[model_name][threshold_type] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'confusion_matrix': confusion_matrix(self.y_val, y_pred).tolist()
            }

            # Matriz de confusión
            cm = confusion_matrix(self.y_val, y_pred)
            print(f"\nMatriz de Confusión:")
            print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
            print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    def plot_results(self):
        """Genera visualizaciones de los resultados del entrenamiento"""
        print("\n" + "=" * 80)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 80)

        for model_name in self.models.keys():
            self._plot_model_results(model_name)

        # Comparación de modelos
        self._plot_model_comparison()

        print("\n✓ Visualizaciones guardadas en", self.results_dir)

    def _plot_model_results(self, model_name):
        """
        Genera visualizaciones para un modelo específico

        Args:
            model_name: Nombre del modelo
        """
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Resultados de Entrenamiento: {model_name.upper()}',
                     fontsize=16, fontweight='bold')

        # 1. Curva ROC
        fpr, tpr, _ = roc_curve(self.y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('Curva ROC')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(self.y_val, y_pred_proba)

        axes[0, 1].plot(recall, precision, color='blue', lw=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Curva Precision-Recall')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Matriz de Confusión (Youden)
        thresholds = self.thresholds[model_name]
        y_pred_youden = (y_pred_proba >= thresholds['youden']).astype(int)
        cm_youden = confusion_matrix(self.y_val, y_pred_youden)

        sns.heatmap(cm_youden, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
        axes[1, 0].set_title(f'Matriz de Confusión (Threshold Youden: {thresholds["youden"]:.4f})')
        axes[1, 0].set_ylabel('Real')
        axes[1, 0].set_xlabel('Predicción')

        # 4. Matriz de Confusión (Probabilístico)
        y_pred_prob = (y_pred_proba >= thresholds['probabilistic']).astype(int)
        cm_prob = confusion_matrix(self.y_val, y_pred_prob)

        sns.heatmap(cm_prob, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                   xticklabels=['No CKD', 'CKD'], yticklabels=['No CKD', 'CKD'])
        axes[1, 1].set_title(f'Matriz de Confusión (Threshold Probabilístico: {thresholds["probabilistic"]:.4f})')
        axes[1, 1].set_ylabel('Real')
        axes[1, 1].set_xlabel('Predicción')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_name}_training_results_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Feature Importance (si aplica)
        if model_name == 'xgboost':
            self._plot_feature_importance(model_name)

    def _plot_feature_importance(self, model_name):
        """
        Genera visualización de importancia de features para XGBoost

        Args:
            model_name: Nombre del modelo
        """
        model = self.models[model_name]

        # Obtener importancia de features
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20

        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Features más importantes - {model_name.upper()}',
                 fontsize=14, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Importancia')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{model_name}_feature_importance_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_comparison(self):
        """Genera visualización comparativa entre modelos"""
        metrics_comparison = {}

        for model_name in self.models.keys():
            for threshold_type in ['youden', 'probabilistic']:
                key = f"{model_name}_{threshold_type}"
                metrics_comparison[key] = self.metrics[model_name][threshold_type]

        # Preparar datos para visualización
        models = list(metrics_comparison.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Comparación de Modelos', fontsize=16, fontweight='bold')

        # Gráfico de barras agrupadas
        x = np.arange(len(metrics_names))
        width = 0.15

        for i, model in enumerate(models):
            values = [metrics_comparison[model][metric] for metric in metrics_names]
            axes[0].bar(x + i * width, values, width, label=model)

        axes[0].set_xlabel('Métricas')
        axes[0].set_ylabel('Valor')
        axes[0].set_title('Comparación de Métricas')
        axes[0].set_xticks(x + width * (len(models) - 1) / 2)
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])

        # Tabla resumen
        axes[1].axis('tight')
        axes[1].axis('off')

        table_data = []
        for model in models:
            row = [model.replace('_', ' ').title()]
            for metric in metrics_names:
                row.append(f"{metrics_comparison[model][metric]:.4f}")
            table_data.append(row)

        headers = ['Modelo'] + [m.replace('_', ' ').title() for m in metrics_names]
        table = axes[1].table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Colorear encabezados
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'model_comparison_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_models(self):
        """Guarda los modelos, preprocessors y metadatos"""
        print("\n" + "=" * 80)
        print("GUARDANDO MODELOS")
        print("=" * 80)

        for model_name, model in self.models.items():
            # Guardar modelo
            model_path = self.weights_dir / f'{model_name}_model_{self.timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Modelo guardado: {model_path}")

        # Guardar preprocessors
        preprocessor_path = self.weights_dir / f'preprocessors_{self.timestamp}.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'imputer': self.imputer,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"✓ Preprocessors guardados: {preprocessor_path}")

        # Guardar thresholds
        thresholds_path = self.weights_dir / f'thresholds_{self.timestamp}.json'
        with open(thresholds_path, 'w') as f:
            json.dump(self.thresholds, f, indent=4)
        print(f"✓ Thresholds guardados: {thresholds_path}")

        # Guardar métricas
        metrics_path = self.results_dir / f'training_metrics_{self.timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"✓ Métricas guardadas: {metrics_path}")

        # Guardar resumen
        summary = {
            'timestamp': self.timestamp,
            'data_shape': self.df.shape,
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'models': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'class_distribution': {
                'no_ckd': int((self.y_train == 0).sum()),
                'ckd': int((self.y_train == 1).sum())
            }
        }

        summary_path = self.results_dir / f'training_summary_{self.timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✓ Resumen guardado: {summary_path}")

        print(f"\n{'=' * 80}")
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Modelos entrenados: {', '.join(self.models.keys())}")
        print(f"Resultados guardados en: {self.results_dir}")
        print(f"Pesos guardados en: {self.weights_dir}")


def main():
    """Función principal para ejecutar el entrenamiento"""
    # Configuración
    DATA_PATH = '../../data/Chronic_Kidney_Dsease_data.csv'
    RESULTS_DIR = 'results'
    WEIGHTS_DIR = '../algorithm_module/weights'

    # Crear entrenador
    trainer = CKDModelTrainer(DATA_PATH, RESULTS_DIR, WEIGHTS_DIR)

    # Pipeline de entrenamiento
    trainer.load_and_prepare_data(test_size=0.2, random_state=42)
    trainer.train_logistic_regression(max_iter=1000)
    trainer.train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
    trainer.plot_results()
    trainer.save_models()


if __name__ == '__main__':
    main()
