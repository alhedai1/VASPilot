from vaspilot.tools.wait_calc_tool import WaitCalcTool


def test_wait_stops_on_bader_monitor_error(monkeypatch):
    calls = []

    async def check(self, calculation_ids):
        calls.append(calculation_ids)
        if len(calls) > 1:
            raise KeyboardInterrupt("wait should stop after terminal error")
        return {"bader-1": {"status": "error", "error": "Bader monitor failed"}}

    monkeypatch.setattr(WaitCalcTool, "_check_status", check)
    monkeypatch.setattr("vaspilot.tools.wait_calc_tool.time.sleep", lambda _: None)
    result = WaitCalcTool("http://unused")._run(["bader-1"])
    assert result["bader-1"]["status"] == "error"
