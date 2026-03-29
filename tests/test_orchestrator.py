import threading
from unittest.mock import patch

from voice_app.state_machine import State


def make_mock_config(tmp_path):
    from voice_app.config import (
        AppConfig,
        AudioConfig,
        DeepgramConfig,
        HistoryConfig,
        LLMConfig,
        TTSConfig,
        VADConfig,
    )

    # aec defaults to AECConfig(enabled=False) — half-duplex path
    return AppConfig(
        system_prompt="You are a test assistant.",
        audio=AudioConfig(),
        vad=VADConfig(),
        deepgram=DeepgramConfig(),
        llm=LLMConfig(),
        tts=TTSConfig(),
        history=HistoryConfig(file=str(tmp_path / "history.json")),
        openai_api_key="sk-test",
        deepgram_api_key="dg-test",
    )


def make_mock_config_aec(tmp_path):
    """Config with aec.enabled = True for testing the AEC code path."""
    from voice_app.config import (
        AECConfig,
        AppConfig,
        AudioConfig,
        DeepgramConfig,
        HistoryConfig,
        LLMConfig,
        TTSConfig,
        VADConfig,
    )

    return AppConfig(
        system_prompt="You are a test assistant.",
        audio=AudioConfig(),
        vad=VADConfig(),
        deepgram=DeepgramConfig(),
        llm=LLMConfig(),
        tts=TTSConfig(),
        history=HistoryConfig(file=str(tmp_path / "history.json")),
        openai_api_key="sk-test",
        deepgram_api_key="dg-test",
        aec=AECConfig(enabled=True),
    )


def make_orchestrator(tmp_path):
    from voice_app.orchestrator import Orchestrator

    cfg = make_mock_config(tmp_path)

    with (
        patch("voice_app.orchestrator.AudioCapture"),
        patch("voice_app.orchestrator.VADDetector"),
        patch("voice_app.orchestrator.STTClient"),
        patch("voice_app.orchestrator.LLMClient"),
        patch("voice_app.orchestrator.TTSEngine"),
    ):
        orch = Orchestrator(cfg, debug=False)
        mock_tts = orch.tts
        mock_llm = orch.llm
        mock_vad = orch.vad

    return orch, mock_tts, mock_llm, mock_vad


def make_orchestrator_aec(tmp_path):
    """
    Build an Orchestrator with aec.enabled = True, mocking AEC components so the
    test suite runs without speexdsp installed.
    """
    from voice_app.orchestrator import Orchestrator

    cfg = make_mock_config_aec(tmp_path)

    # Patch AECProcessor and SpeakerReferenceBuffer at the module level so the
    # local `from voice_app.aec import ...` inside Orchestrator.__init__ picks up
    # the mocks rather than trying to import speexdsp.
    with (
        patch("voice_app.aec.AECProcessor"),
        patch("voice_app.aec.SpeakerReferenceBuffer"),
        patch("voice_app.orchestrator.AudioCapture"),
        patch("voice_app.orchestrator.VADDetector"),
        patch("voice_app.orchestrator.STTClient"),
        patch("voice_app.orchestrator.LLMClient"),
        patch("voice_app.orchestrator.TTSEngine"),
    ):
        orch = Orchestrator(cfg, debug=False)
        mock_stt = orch.stt
        mock_tts = orch.tts
        mock_llm = orch.llm
        mock_vad = orch.vad

    return orch, mock_stt, mock_tts, mock_llm, mock_vad


def pump_events(orch, *events):
    for event in events:
        orch.event_queue.put(event)
    orch.event_queue.put({"type": "SHUTDOWN"})

    thread = threading.Thread(target=orch._event_loop)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), "Event loop did not exit in time"


class TestStateMachineIntegration:
    def test_speech_detected_moves_to_listening(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        assert orch.state_machine.state == State.IDLE
        pump_events(orch, {"type": "SPEECH_DETECTED"})
        assert orch.state_machine.state == State.LISTENING

    def test_utterance_complete_moves_to_processing(self, tmp_path):
        orch, _, mock_llm, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.LISTENING
        pump_events(orch, {"type": "UTTERANCE_COMPLETE", "text": "Hello"})
        assert orch.state_machine.state == State.PROCESSING
        mock_llm.submit.assert_called_once()

    def test_llm_response_ready_does_not_leave_speaking_early(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.PROCESSING
        orch._gen_id = 3
        pump_events(
            orch,
            {"type": "FIRST_TTS_CHUNK"},
            {"type": "LLM_RESPONSE_READY", "gen_id": 3, "response": "Hi!"},
        )
        assert orch.state_machine.state == State.SPEAKING

    def test_tts_complete_moves_to_idle(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 3
        orch._pending_response_by_gen[3] = "Hi!"
        pump_events(orch, {"type": "TTS_COMPLETE", "gen_id": 3})
        assert orch.state_machine.state == State.IDLE

    def test_stale_tts_complete_ignored(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 5
        pump_events(orch, {"type": "TTS_COMPLETE", "gen_id": 3})
        assert orch.state_machine.state == State.SPEAKING


class TestInterruptPath:
    def test_interrupt_during_speaking_calls_tts_and_llm(self, tmp_path):
        orch, mock_tts, mock_llm, mock_vad = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        pump_events(orch, {"type": "INTERRUPT"})

        mock_tts.interrupt.assert_called_once()
        mock_llm.cancel.assert_called_once()
        mock_vad.reset.assert_called_once()
        assert orch.state_machine.state == State.LISTENING

    def test_interrupt_during_processing_also_works(self, tmp_path):
        orch, mock_tts, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.PROCESSING
        pump_events(orch, {"type": "INTERRUPT"})

        mock_tts.interrupt.assert_called_once()
        assert orch.state_machine.state == State.LISTENING

    def test_interrupt_bumps_gen_id(self, tmp_path):
        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 7
        pump_events(orch, {"type": "INTERRUPT"})
        assert orch._gen_id == 8

    def test_interrupt_from_idle_is_ignored(self, tmp_path):
        orch, mock_tts, _, _ = make_orchestrator(tmp_path)
        assert orch.state_machine.state == State.IDLE
        pump_events(orch, {"type": "INTERRUPT"})
        mock_tts.interrupt.assert_not_called()
        assert orch.state_machine.state == State.IDLE


class TestHistoryPersistence:
    def test_assistant_response_persisted_on_tts_complete(self, tmp_path):
        from voice_app.history import load_history

        orch, _, _, _ = make_orchestrator(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 1
        pump_events(
            orch,
            {"type": "LLM_RESPONSE_READY", "gen_id": 1, "response": "Hey! I'm doing great."},
            {"type": "TTS_COMPLETE", "gen_id": 1},
        )
        messages = load_history(str(tmp_path / "history.json"))
        assert any(
            message["role"] == "assistant" and "great" in message["content"]
            for message in messages
        )


class TestAECOrchestrator:
    """
    Verify the AEC code path in the orchestrator.

    With aec.enabled = True:
      • FIRST_TTS_CHUNK must NOT call stt.set_echo_suppression — AEC removes
        the echo upstream so half-duplex muting is unnecessary.
      • INTERRUPT must NOT call stt.set_echo_suppression — same reason.
      • TTS_COMPLETE must NOT schedule the echo-release timer.
    """

    def test_first_tts_chunk_does_not_suppress_stt_with_aec(self, tmp_path):
        """AEC path: FIRST_TTS_CHUNK skips STT silence injection."""
        orch, mock_stt, _, _, _ = make_orchestrator_aec(tmp_path)
        orch.state_machine.state = State.PROCESSING
        pump_events(orch, {"type": "FIRST_TTS_CHUNK"})

        mock_stt.set_echo_suppression.assert_not_called()
        assert orch.state_machine.state == State.SPEAKING

    def test_interrupt_does_not_suppress_stt_with_aec(self, tmp_path):
        """AEC path: INTERRUPT does not call STT echo suppression helpers."""
        orch, mock_stt, _, _, _ = make_orchestrator_aec(tmp_path)
        orch.state_machine.state = State.SPEAKING
        pump_events(orch, {"type": "INTERRUPT"})

        mock_stt.set_echo_suppression.assert_not_called()
        assert orch.state_machine.state == State.LISTENING

    def test_tts_complete_does_not_schedule_echo_release_with_aec(self, tmp_path):
        """AEC path: TTS_COMPLETE leaves echo-suppress timer as None."""
        orch, _, _, _, _ = make_orchestrator_aec(tmp_path)
        orch.state_machine.state = State.SPEAKING
        orch._gen_id = 1
        pump_events(orch, {"type": "TTS_COMPLETE", "gen_id": 1})

        assert orch._echo_suppress_timer is None
        assert orch.state_machine.state == State.IDLE

    def test_echo_interim_does_not_interrupt_with_aec(self, tmp_path):
        orch, _, mock_tts, mock_llm, _ = make_orchestrator_aec(tmp_path)
        orch._gen_id = 1
        orch.state_machine.state = State.SPEAKING

        pump_events(
            orch,
            {"type": "ASSISTANT_RESPONSE_CHUNK", "gen_id": 1, "text": "Let me explain the plan."},
            {"type": "TRANSCRIPT_INTERIM", "text": "explain the plan"},
        )

        mock_tts.interrupt.assert_not_called()
        mock_llm.cancel.assert_not_called()
        assert orch.state_machine.state == State.SPEAKING

    def test_non_echo_interim_interrupts_with_aec(self, tmp_path):
        orch, _, mock_tts, mock_llm, mock_vad = make_orchestrator_aec(tmp_path)
        orch._gen_id = 1
        orch.state_machine.state = State.SPEAKING

        pump_events(
            orch,
            {"type": "ASSISTANT_RESPONSE_CHUNK", "gen_id": 1, "text": "Let me explain the plan."},
            {"type": "TRANSCRIPT_INTERIM", "text": "stop and listen"},
        )

        mock_tts.interrupt.assert_called_once()
        mock_llm.cancel.assert_called_once()
        mock_vad.reset.assert_called_once()
        assert orch.state_machine.state == State.LISTENING

    def test_echo_utterance_after_tts_complete_is_ignored(self, tmp_path):
        orch, _, _, mock_llm, _ = make_orchestrator_aec(tmp_path)
        orch._gen_id = 1
        orch.state_machine.state = State.SPEAKING
        orch._pending_response_by_gen[1] = "Let me explain the plan."

        pump_events(
            orch,
            {"type": "ASSISTANT_RESPONSE_CHUNK", "gen_id": 1, "text": "Let me explain the plan."},
            {"type": "TTS_COMPLETE", "gen_id": 1},
            {"type": "UTTERANCE_COMPLETE", "text": "explain the plan"},
        )

        mock_llm.submit.assert_not_called()
        assert orch.state_machine.state == State.IDLE
