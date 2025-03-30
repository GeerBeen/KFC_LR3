document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("weatherForm");
    const predictionElement = document.getElementById("prediction");

    form.addEventListener("submit", async function (event) {
        event.preventDefault(); 

        const weatherData = {
            temperature_lag1: parseFloat(document.getElementById("temperature_lag1").value),
            temperature_lag2: parseFloat(document.getElementById("temperature_lag2").value),
            temperature_lag3: parseFloat(document.getElementById("temperature_lag3").value),
            temperature_lag4: parseFloat(document.getElementById("temperature_lag4").value),
            precipitation: parseFloat(document.getElementById("precipitation").value),
            humidity: parseFloat(document.getElementById("humidity").value),
            wind_speed: parseFloat(document.getElementById("wind_speed").value),
            cloud_cover: parseFloat(document.getElementById("cloud_cover").value),
            day_of_year: parseInt(document.getElementById("day_of_year").value),
            pressure: parseFloat(document.getElementById("pressure").value)
        };

        try {
            const response = await fetch("http://127.0.0.1:8000/next_day_temperature/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(weatherData)
            });

            if (!response.ok) {
                throw new Error("Помилка при отриманні прогнозу");
            }

            const data = await response.json();
            predictionElement.textContent = data.predicted_temperature.toFixed(2);
        } catch (error) {
            console.error("Помилка:", error);
            predictionElement.textContent = "Помилка!";
        }
    });
});
