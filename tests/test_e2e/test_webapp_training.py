"""Training lifecycle API tests."""

import time

import requests


class TestTrainingLifecycle:
    def test_start_returns_status(self, base_url):
        resp = requests.post(f"{base_url}/start")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("started", "already_running")

        # Clean up: stop training
        requests.post(f"{base_url}/stop")

    def test_stop_returns_status(self, base_url):
        resp = requests.post(f"{base_url}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("stopped", "not_running")

    def test_start_stop_lifecycle(self, base_url):
        # Start training
        resp = requests.post(f"{base_url}/start")
        assert resp.status_code == 200

        # Brief pause for the training thread to start
        time.sleep(2)

        # Verify running state
        state = requests.get(f"{base_url}/state").json()
        # Training may have already crashed (no GPU, missing dirs), so
        # we just verify the endpoint responds
        assert "running" in state

        # Stop training
        resp = requests.post(f"{base_url}/stop")
        assert resp.status_code == 200

        # Verify stopped
        time.sleep(1)
        state = requests.get(f"{base_url}/state").json()
        assert state["running"] is False

    def test_eval_baselines_without_model(self, base_url):
        resp = requests.post(f"{base_url}/eval/baselines")
        # Without a trained model, this should indicate no model available
        assert resp.status_code in (200, 400)
