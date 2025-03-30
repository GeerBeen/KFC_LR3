from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("ml/random_forest_model.pkl")


class WeatherInfo(BaseModel):
    temperature_lag1: float = Field(15.0, ge=-100, le=100, description="Температура за 1 день до поточного °C")
    temperature_lag2: float = Field(14.8, ge=-100, le=100, description="Температура за 2 дні до поточного °C")
    temperature_lag3: float = Field(14.5, ge=-100, le=100, description="Температура за 3 дні до поточного °C")
    temperature_lag4: float = Field(14.2, ge=-100, le=100, description="Температура за 4 дні до поточного °C")
    precipitation: float = Field(5.0, ge=0, le=1000, description="Сумарні опади минулого дня мм")
    humidity: float = Field(65.0, ge=0, le=100, description="Середня вологість минулого дня %")
    wind_speed: float = Field(12.3, ge=0, le=200, description="Середня швидкість вітру минулого дня км/год")
    cloud_cover: float = Field(80.0, ge=0, le=100, description="Середня загальна минулого дня хмарність %")
    day_of_year: int = Field(150, ge=1, le=366, description="Номер минулого дня у році")
    pressure: float = Field(1012.5, ge=850, le=1085,
                            description="Середній атмосферний тиск на рівні моря минулого дня hPa")


class ForecastResponse(BaseModel):
    predicted_temperature: float = Field(..., description="Прогнозована температура на завтра °C")


def fake_foresight(weather: WeatherInfo):
    return weather.mean_temperature


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/next_day_temperature/", response_model=ForecastResponse)
def make_foresight(weather: WeatherInfo):
    input_data = np.array([[weather.precipitation, weather.humidity, weather.wind_speed,
                            weather.cloud_cover, weather.pressure, weather.temperature_lag1, weather.temperature_lag2,
                            weather.temperature_lag3,
                            weather.temperature_lag4, weather.day_of_year]])

    predicted_temp = model.predict(input_data)[0]
    return ForecastResponse(predicted_temperature=predicted_temp)


@app.get("/health")
def health_check():
    try:
        test_data = [[15.0, 14.8, 14.5, 14.2, 5.0, 65.0, 12.3, 80.0, 150, 1012.5]]
        model.predict(test_data)
        return {"status": "ok", "message": "API та модель працюють"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)
