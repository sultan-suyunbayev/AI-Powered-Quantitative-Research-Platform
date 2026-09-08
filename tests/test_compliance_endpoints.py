# -*- coding: utf-8 -*-
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-compliance")

import app as app_module
from app import api

# The global auth middleware only whitelists loopback peers; the TestClient
# peer is "testclient", so authenticate explicitly with the API token.
client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})


def test_compliance_clock_status():
    response = client.get("/api/compliance/clock/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "drift_microseconds" in data
    assert "severity" in data
    assert "rts25_compliant" in data


def test_compliance_conformance_run():
    response = client.post("/api/compliance/conformance/run", json={"algo_id": "test_strategy"})
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "status" in data
    assert "tests" in data
    assert len(data["tests"]) > 0


def test_compliance_best_execution_report():
    response = client.get("/api/compliance/best-execution/report")
    assert response.status_code == 200
    data = response.json()
    assert "venues" in data
    assert "policy_version" in data
    assert "policy_hash" in data
    assert len(data["venues"]) > 0


def test_dora_incident_report():
    payload = {
        "title": "Database Outage Test",
        "description": "ICT database synchronization loss",
        "financial_impact_eur": 50000.0,
        "duration_minutes": 25.0,
        "clients_affected": 300,
        "data_loss_type": "trading",
    }
    response = client.post("/api/dora/incidents/report", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "assessment" in data
    assert "is_major" in data
    assert "report" in data


def test_dora_concentration_risk():
    response = client.get("/api/dora/concentration-risk")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "metrics" in data
    assert "risks" in data


def test_dora_roi_generate():
    response = client.post("/api/dora/roi/generate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "xml_report_path" in data
    assert "json_report_path" in data


def test_dora_bcp_simulate():
    payload = {"scenario": "Primary DB Corruption"}
    response = client.post("/api/dora/bcp/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "activated"
    assert data["scenario"] == "Primary DB Corruption"
    assert "steps" in data


def test_ai_act_explain_recent():
    response = client.get("/api/ai-act/explain/recent")
    assert response.status_code == 200
    data = response.json()
    assert "explanations" in data
    assert "stats" in data


def test_ai_act_explain_tx():
    # DEC-8001 is one of the decisions the app records at start-up.
    response = client.get("/api/ai-act/explain/DEC-8001")
    assert response.status_code == 200
    data = response.json()
    assert data["decision_id"] == "DEC-8001"
    assert "feature_importance" in data
    assert "rational_explanation" in data


def test_ai_act_explain_unknown_tx_is_not_synthesised():
    """An explainability record must map to a real logged decision."""
    response = client.get("/api/ai-act/explain/TX-DOES-NOT-EXIST")
    assert response.status_code == 404


def test_ai_act_veto_override():
    response = client.post("/api/ai-act/oversight/veto", json={"veto_active": True})
    assert response.status_code == 200
    data = response.json()
    assert data["veto_active"] is True
    assert "disarmed" in data["message"].lower()

    # Reset
    response_reset = client.post("/api/ai-act/oversight/veto", json={"veto_active": False})
    assert response_reset.status_code == 200
    assert response_reset.json()["veto_active"] is False


def test_ai_act_conformity_status():
    response = client.get("/api/ai-act/conformity/status")
    assert response.status_code == 200
    data = response.json()
    assert data["risk_management_system"] == "implemented"
    assert data["conformity_declaration_issued"] is True


def test_gdpr_export():
    response = client.post("/api/gdpr/export", json={"client_id": "test_client"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "download_url" in data


def test_gdpr_delete():
    response = client.post("/api/gdpr/delete", json={"client_id": "test_client"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "anonymized" in data["message"].lower()


def test_retention_hold_and_ledger():
    # Toggle Legal Hold
    response = client.post("/api/compliance/retention/hold", json={"active": True})
    assert response.status_code == 200
    data = response.json()
    assert data["legal_hold_active"] is True

    # Get ledger and confirm hold is applied to volume
    ledger_resp = client.get("/api/compliance/retention/ledger")
    assert ledger_resp.status_code == 200
    volumes = ledger_resp.json()
    assert len(volumes) > 0
    # Last volume should have legal_hold = True
    assert volumes[-1]["legal_hold"] is True

    # Release Legal Hold
    response_release = client.post("/api/compliance/retention/hold", json={"active": False})
    assert response_release.status_code == 200
    assert response_release.json()["legal_hold_active"] is False


def test_surveillance_otr():
    response = client.get("/api/compliance/surveillance/otr")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert "venue" in data[0]
    assert "otr_volume_ratio" in data[0]


def test_pre_trade_limits():
    payload = {
        "max_order_value": 2000000.0,
        "max_order_volume": 20000.0,
        "price_collar_pct": 3.5,
        "daily_loss_limit": 40000.0,
    }
    response = client.post("/api/compliance/risk/pre-trade/update", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["max_order_value"] == 2000000.0
    assert data["price_collar_pct"] == 3.5

    # Fetch and check
    limits_resp = client.get("/api/compliance/risk/pre-trade/limits")
    assert limits_resp.status_code == 200
    limits_data = limits_resp.json()
    assert "max_order_value" in limits_data


def test_killswitch_trigger():
    payload = {"scope": "XLON", "reason": "Test surveillance breach"}
    response = client.post("/api/compliance/killswitch/trigger", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["scope"] == "XLON"
    assert data["tripped"] is True
    # The kill switch halts new order flow; cancelling resting orders is
    # broker/OMS-side and asynchronous, so no cancellation count is reported.
    assert "cancelled_orders_count" not in data
    assert "message" in data
