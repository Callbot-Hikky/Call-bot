import pytest

from hikky.adapters.back.http_client import BackHttpClient

BASE_URL = "https://back.test.hikky.example"
API_KEY = "test-key"


@pytest.fixture
def back_client() -> BackHttpClient:
    return BackHttpClient(base_url=BASE_URL, api_key=API_KEY)


@pytest.fixture
def base_url() -> str:
    return BASE_URL
