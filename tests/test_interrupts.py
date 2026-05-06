from __future__ import annotations

import pytest


def test_first_ctrl_c_raises_keyboard_interrupt(monkeypatch, capsys):
    import cuda_opt_agent.interrupts as interrupts

    interrupts._reset_interrupt_state_for_tests()
    monkeypatch.setattr(interrupts.time, "monotonic", lambda: 10.0)

    with pytest.raises(KeyboardInterrupt):
        interrupts._handle_sigint(2, None)

    captured = capsys.readouterr()
    assert "saving state" in captured.err


def test_second_ctrl_c_force_exits(monkeypatch, capsys):
    import cuda_opt_agent.interrupts as interrupts

    interrupts._reset_interrupt_state_for_tests()
    times = iter([10.0, 11.0])
    exits = []
    monkeypatch.setattr(interrupts.time, "monotonic", lambda: next(times))

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(interrupts.os, "_exit", fake_exit)

    with pytest.raises(KeyboardInterrupt):
        interrupts._handle_sigint(2, None)
    with pytest.raises(SystemExit):
        interrupts._handle_sigint(2, None)

    captured = capsys.readouterr()
    assert exits == [130]
    assert "Force exit" in captured.err
