import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health_check():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validation():
    data = {
        "temperature_lag1": "str",
        "temperature_lag2": 14.8,
        "temperature_lag3": 14.5,
        "temperature_lag4": 14.2,
        "precipitation": 5.0,
        "humidity": 65.0,
        "wind_speed": 12.3,
        "cloud_cover": 80.0,
        "day_of_year": 150,
        "pressure": 999999
    }
    response = requests.post(f"{BASE_URL}/next_day_temperature/", json=data)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_prediction():
    data = {
        "temperature_lag1": 15.0,
        "temperature_lag2": 14.8,
        "temperature_lag3": 14.5,
        "temperature_lag4": 14.2,
        "precipitation": 5.0,
        "humidity": 65.0,
        "wind_speed": 12.3,
        "cloud_cover": 80.0,
        "day_of_year": 150,
        "pressure": 1012.5
    }
    response = requests.post(f"{BASE_URL}/next_day_temperature/", json=data)
    assert response.status_code == 200
    assert -50 <= response.json()["predicted_temperature"] <= 50


if __name__ == '__main__':
    test_health_check()
    test_validation()
    test_prediction()
