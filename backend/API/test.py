import pytest
import httpx
from fastapi import status

BASE_URL = "http://localhost:8000"

@pytest.mark.asyncio
async def test_register_patient():
    async with httpx.AsyncClient() as client:
        payload = {
            "name": "Test User",
            "birthdate": "01/01/1990",
            "gender": "male",
            "work": "Tester",
            "username": "testuser",
            "email": "test@example.com",
            "phone": "1234567890",
            "password": "testpassword123"
        }
        response = await client.post(f"{BASE_URL}/register", json=payload)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["username"] == "testuser"
        assert "access_token" in response.json()

@pytest.mark.asyncio
async def test_login():
    async with httpx.AsyncClient() as client:
        payload = {
            "username": "testuser",
            "password": "testpassword123"
        }
        response = await client.post(f"{BASE_URL}/login", data=payload)
        assert response.status_code == status.HTTP_200_OK
        assert "access_token" in response.json()

@pytest.mark.asyncio
async def test_get_current_user():
    async with httpx.AsyncClient() as client:
        # Đăng nhập để lấy token
        login_payload = {"username": "testuser", "password": "testpassword123"}
        login_response = await client.post(f"{BASE_URL}/login", data=login_payload)
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{BASE_URL}/me", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["username"] == "testuser"

# Thêm các test khác cho admin, medical history, v.v.