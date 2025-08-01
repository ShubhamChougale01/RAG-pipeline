import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_weather(query: str) -> str:
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        city = query.split("weather in")[-1].strip()
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return f"Weather in {city}: {data['weather'][0]['description']}, Temperature: {data['main']['temp']}°C"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"