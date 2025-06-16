import matplotlib.pyplot as plt
import seaborn as sns
from utils import FEATURES, TARGET


def analyze_data(df, save_path="reports"):
    """Анализ данных и построение графиков"""
    print("[INFO] Анализ данных...")
    print("Баланс классов:")
    print(df[TARGET].value_counts())

    # График баланса классов
    plt.figure(figsize=(8, 6))
    sns.countplot(
        x=TARGET,
        data=df
    )
    plt.title("Распределение целевой переменной")
    plt.savefig(f"{save_path}/target_distribution.png")
    plt.close()

    # Корреляция с целевой переменной
    correlation = df[FEATURES + [TARGET]].corr()[[TARGET]]
    sorted_corr = correlation.sort_values(by='target')

    plt.figure(figsize=(10, 8))
    sorted_corr.plot(kind='barh')
    plt.title("Корреляция признаков с целевой переменной")
    plt.savefig(f"{save_path}/features_correlation.png")
    plt.close()
