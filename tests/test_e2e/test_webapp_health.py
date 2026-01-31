"""API endpoint tests for the webapp (no browser needed)."""

import requests


class TestHealthEndpoint:
    def test_health_returns_ok(self, base_url):
        resp = requests.get(f"{base_url}/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "ok"
        assert data["training"] is False
        assert "uptime_seconds" in data
        assert "version" in data

    def test_state_endpoint(self, base_url):
        resp = requests.get(f"{base_url}/state")
        assert resp.status_code == 200

        data = resp.json()
        assert "running" in data
        assert "epoch" in data
        assert "loss" in data
        assert data["running"] is False

    def test_models_list_endpoint(self, base_url):
        resp = requests.get(f"{base_url}/models/list")
        assert resp.status_code == 200

        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
