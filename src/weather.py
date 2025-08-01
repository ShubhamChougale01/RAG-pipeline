import requests
import os

def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return f"Weather in {city}: {data['weather'][0]['description']}, Temperature: {data['main']['temp']}°C"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"