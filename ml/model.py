import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib


def add_lag_features(df, target_col="temperature", max_lag=3):
    df = df.copy()
    for lag in range(1, max_lag + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df.dropna()


def get_data_df_and_prepare(file: str):
    df = pd.read_csv(file, parse_dates=["timestamp"])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = add_lag_features(df, max_lag=4)

    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["target_temperature"] = df["temperature"].shift(-1)

    df = df.dropna()
    return df


def teach_model(df: pd.DataFrame):
    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    X_train = train.drop(columns=["timestamp", "temperature", "target_temperature"])
    y_train = train["target_temperature"]

    X_test = test.drop(columns=["timestamp", "temperature", "target_temperature"])
    y_test = test["target_temperature"]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    """
    # model = LinearRegression()
    model = RandomForestRegressor(random_state=42)
    # model = XGBRegressor(objective="reg:squarederror", random_state=4)

    model.fit(X_train, y_train)
    test_model(model, X_test, y_test)
    return model


def test_model(model, x_test, y_test):
    importances = model.feature_importances_
    feature_names = model.feature_names_in_

    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}°C")
    print(f"Mean Squared Error: {mse:.2f}")


def make_and_savemodel(from_file: str, to_file: str):
    proceeded_df = get_data_df_and_prepare(from_file)
    model = teach_model(proceeded_df)
    joblib.dump(model, to_file)


if __name__ == '__main__':
    make_and_savemodel('clean_data.csv', "random_forest_model_test.pkl")
