import pytest
from src.weather import get_weather
import os

def test_get_weather_success(monkeypatch):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
        def json(self):
            return {"weather": [{"description": "clear sky"}], "main": {"temp": 25}}
        def raise_for_status(self):
            pass

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)
    result = get_weather("London")
    assert "Weather in London: clear sky, Temperature: 25°C" in result

def test_get_weather_failure(monkeypatch):
    class MockResponse:
        def __init__(self):
            self.status_code = 404
        def raise_for_status(self):
            raise requests.exceptions.RequestException("City not found")

    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)
    result = get_weather("InvalidCity")
    assert "Error fetching weather data" in result