import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

def get_weather(query: str) -> str:
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return "Error: OPENWEATHERMAP_API_KEY not found in environment variables."

        match = re.search(r"weather\s+(?:in|for)?\s+([a-zA-Z\s,]+?)(?:\?|$)", query, re.IGNORECASE)
        if not match:
            return "Error: Could not extract city name from query."
        city = match.group(1).strip()

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()

        return f"Weather in {city}: {data['weather'][0]['description']}, Temperature: {data['main']['temp']}°C"
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401:
            return "Error: Invalid or unauthorized OpenWeatherMap API key."
        elif response.status_code == 404:
            return f"Error: City '{city}' not found or invalid API request."
        return f"Error fetching weather: {str(http_err)}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"