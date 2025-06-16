import argparse
from data_loader import load_data
from data_analyzer import analyze_data
from model_trainer import train_model, evaluate_model
from model_predictor import predict_new_data, evaluate_model_from_file
from sklearn.model_selection import train_test_split
from utils import FEATURES, TARGET


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Scoring Tool")
    subparsers = parser.add_subparsers(dest='command')

    # Команда: analyze
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Анализ данных'
    )
    analyze_parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Путь к dataset.parquet'
    )

    # Команда: train
    train_parser = subparsers.add_parser(
        'train',
        help='Обучение модели'
    )
    train_parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Путь к dataset.parquet'
    )

    # Команда: predict
    predict_parser = subparsers.add_parser(
        'predict',
        help='Предсказание на новых данных'
    )
    predict_parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Путь к test_data.parquet'
    )

    args = parser.parse_args()

    if args.command == 'analyze':
        df = load_data(args.data_path)
        analyze_data(df)

    elif args.command == 'train':
        df = load_data(args.data_path)
        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

    elif args.command == 'predict':
        predict_new_data(args.data_path)

    elif args.command == 'evaluate':
        evaluate_model_from_file(args.data_path)
