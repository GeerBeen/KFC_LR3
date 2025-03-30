import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', None)


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith("timestamp"):
                    skip_rows = i
                    break
            else:
                raise ValueError("Не знайдено заголовок 'timestamp' у файлі.")

        df = pd.read_csv(file_path, skiprows=skip_rows)

        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        df.columns = df.columns.str.replace(r"^Basel ", "", regex=True)

        rename_dict = {
            "Temperature [2 m elevation corrected].2": "temperature",
            "Precipitation Total": "precipitation",
            "Relative Humidity [2 m].2": "humidity",
            "Wind Speed [10 m].2": "wind_speed",
            "Cloud Cover Total": "cloud_cover",
            "Mean Sea Level Pressure [MSL].2": "pressure"
        }
        df.rename(columns=rename_dict, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M", errors="coerce")

        df = df.dropna(subset=["timestamp"])

        columns_to_remove = ["Basel Wind Direction [10 m]", "Temperature [2 m elevation corrected]",
                             "Temperature [2 m elevation corrected].1", "Relative Humidity [2 m]",
                             "Relative Humidity [2 m].1", "Wind Speed [10 m]", "Wind Speed [10 m].1",
                             "Wind Direction Dominant [10 m]", "Mean Sea Level Pressure [MSL]",
                             "Mean Sea Level Pressure [MSL].1"]
        df = df.drop(columns=columns_to_remove, errors="ignore")

        numeric_columns = df.columns[1:]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        df = df.dropna()

        return df
    except Exception as e:
        print(f"Помилка при обробці файлу: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    df = load_and_clean_data("dataexport_20250327T144847.csv")
    df.to_csv("clean_data2.csv")
