"""Playwright browser tests for the webapp dashboard UI."""

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture
def dashboard(page: Page, webapp_server):
    """Navigate to the dashboard and wait for it to load."""
    page.goto(webapp_server)
    page.wait_for_load_state("networkidle")
    return page


class TestDashboardLayout:
    def test_page_loads(self, dashboard):
        expect(dashboard.locator("h1")).to_contain_text("Aion-26")

    def test_stat_cards_visible(self, dashboard):
        for card_id in ["win-rate", "samples-sec", "total-samples", "epoch", "loss", "batch-size"]:
            expect(dashboard.locator(f"#{card_id}")).to_be_visible()

    def test_charts_rendered(self, dashboard):
        for chart_id in ["winRateChart", "throughputChart", "lossChart"]:
            expect(dashboard.locator(f"#{chart_id}")).to_be_visible()

    def test_control_buttons(self, dashboard):
        expect(dashboard.get_by_text("Start Training")).to_be_visible()
        expect(dashboard.get_by_text("Stop Training")).to_be_visible()
        expect(dashboard.get_by_text("Run Baselines")).to_be_visible()
        expect(dashboard.get_by_text("Save Model")).to_be_visible()

    def test_status_shows_idle(self, dashboard):
        expect(dashboard.locator("#status")).to_contain_text("IDLE")

    def test_action_distribution_bars(self, dashboard):
        for bar_id in ["fold-bar", "call-bar", "raise-bar", "allin-bar"]:
            expect(dashboard.locator(f"#{bar_id}")).to_be_attached()

    def test_strategy_samples_container(self, dashboard):
        expect(dashboard.locator("#strategy-samples")).to_be_attached()
