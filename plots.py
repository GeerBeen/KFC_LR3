import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv("ml/clean_data.csv", parse_dates=["timestamp"])
    df = df.drop(columns=['Unnamed: 0'])
    df.hist(figsize=(12, 8), bins=50)
    plt.suptitle("Розподіл змінних")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["temperature"], label="Температура")
    plt.xlabel("Дата")
    plt.ylabel("Температура (°C)")
    plt.title("Зміна температури з часом")
    plt.legend()
    plt.show()

    model = joblib.load('ml/random_forest_model.pkl')
    importances = model.feature_importances_
    feature_names = model.feature_names_in_

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color="skyblue")
    plt.xlabel("Важливість ознаки")
    plt.ylabel("Ознаки")
    plt.title("Важливість ознак у моделі")
    plt.gca().invert_yaxis()
    plt.show()
