"""
Script principal para entrenar y evaluar modelos de predicción de CKD
Uso:
    python main.py              # Solo entrena
    python main.py --test       # Entrena y evalúa en test
    python main.py --only-test  # Solo evalúa en test (usa modelo guardado)
"""

import argparse
import sys
from pathlib import Path

# Agregar el directorio Models al path si es necesario
sys.path.append(str(Path(__file__).parent))

from train import CKDModelTrainer
from AIMS_BI_HACKATHON.src.models.test import CKDModelTester


def train_models(data_path, results_dir='results', weights_dir='weights'):
    """Entrena los modelos"""
    print("\n🚀 INICIANDO ENTRENAMIENTO\n")

    trainer = CKDModelTrainer(data_path, results_dir, weights_dir)
    trainer.load_and_prepare_data(test_size=0.2, val_size=0.1, random_state=42)
    trainer.train_logistic_regression(max_iter=1000)
    trainer.train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
    trainer.plot_results()
    trainer.save_models()

    return trainer.timestamp


def test_models(timestamp=None, weights_dir='weights', results_dir='results'):
    """Evalúa los modelos en el conjunto de test"""
    print("\n🧪 INICIANDO EVALUACIÓN EN TEST\n")

    tester = CKDModelTester(
        weights_dir=weights_dir,
        results_dir=results_dir,
        timestamp=timestamp
    )
    tester.load_test_data()
    tester.load_models()
    tester.preprocess_test_data()
    tester.evaluate_all_models()


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar y evaluar modelos de predicción de CKD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                    # Solo entrenar
  python main.py --test             # Entrenar y evaluar
  python main.py --only-test        # Solo evaluar con último modelo
  python main.py --only-test --timestamp 20240115_143022  # Evaluar modelo específico
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Evaluar en test después de entrenar'
    )

    parser.add_argument(
        '--only-test',
        action='store_true',
        help='Solo evaluar en test (sin entrenar)'
    )

    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Timestamp del modelo a evaluar (formato: YYYYMMDD_HHMMSS)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='../data/Chronic_Kidney_Dsease_data.csv',
        help='Ruta al archivo de datos (default: ../data/Chronic_Kidney_Dsease_data.csv)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directorio para resultados (default: results)'
    )

    parser.add_argument(
        '--weights-dir',
        type=str,
        default='weights',
        help='Directorio para pesos (default: weights)'
    )

    args = parser.parse_args()

    # Validar argumentos
    if args.only_test and args.test:
        print("⚠️ Error: No puedes usar --test y --only-test juntos")
        sys.exit(1)

    # Ejecutar según flags
    if args.only_test:
        # Solo evaluar
        test_models(
            timestamp=args.timestamp,
            weights_dir=args.weights_dir,
            results_dir=args.results_dir
        )
    else:
        # Entrenar (siempre)
        timestamp = train_models(
            data_path=args.data,
            results_dir=args.results_dir,
            weights_dir=args.weights_dir
        )

        # Evaluar solo si se indica
        if args.test:
            test_models(
                timestamp=timestamp,
                weights_dir=args.weights_dir,
                results_dir=args.results_dir
            )

    print("\n✅ PROCESO COMPLETADO\n")


if __name__ == '__main__':
    main()