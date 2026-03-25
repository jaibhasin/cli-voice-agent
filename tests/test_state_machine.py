from voice_app.state_machine import AppEvent, State, StateMachine


class TestValidTransitions:
    def test_idle_speech_detected_goes_to_listening(self):
        sm = StateMachine()
        assert sm.state == State.IDLE
        result = sm.transition(AppEvent.SPEECH_DETECTED)
        assert result == State.LISTENING
        assert sm.state == State.LISTENING

    def test_listening_utterance_complete_goes_to_processing(self):
        sm = StateMachine()
        sm.state = State.LISTENING
        result = sm.transition(AppEvent.UTTERANCE_COMPLETE)
        assert result == State.PROCESSING
        assert sm.state == State.PROCESSING

    def test_processing_first_tts_chunk_goes_to_speaking(self):
        sm = StateMachine()
        sm.state = State.PROCESSING
        result = sm.transition(AppEvent.FIRST_TTS_CHUNK)
        assert result == State.SPEAKING
        assert sm.state == State.SPEAKING

    def test_speaking_tts_complete_goes_to_idle(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.TTS_COMPLETE)
        assert result == State.IDLE
        assert sm.state == State.IDLE

    def test_speaking_interrupt_goes_to_listening(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.INTERRUPT)
        assert result == State.LISTENING
        assert sm.state == State.LISTENING

    def test_processing_interrupt_goes_to_listening(self):
        sm = StateMachine()
        sm.state = State.PROCESSING
        result = sm.transition(AppEvent.INTERRUPT)
        assert result == State.LISTENING
        assert sm.state == State.LISTENING


class TestErrorTransition:
    def test_error_from_idle(self):
        sm = StateMachine()
        sm.state = State.IDLE
        result = sm.transition(AppEvent.ERROR)
        assert result == State.IDLE

    def test_error_from_speaking(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.ERROR)
        assert result == State.IDLE
        assert sm.state == State.IDLE

    def test_error_from_processing(self):
        sm = StateMachine()
        sm.state = State.PROCESSING
        result = sm.transition(AppEvent.ERROR)
        assert result == State.IDLE


class TestShutdownTransition:
    def test_shutdown_returns_none(self):
        sm = StateMachine()
        result = sm.transition(AppEvent.SHUTDOWN)
        assert result is None

    def test_shutdown_from_speaking(self):
        sm = StateMachine()
        sm.state = State.SPEAKING
        result = sm.transition(AppEvent.SHUTDOWN)
        assert result is None


class TestIgnoredTransitions:
    def test_idle_ignores_utterance_complete(self):
        sm = StateMachine()
        result = sm.transition(AppEvent.UTTERANCE_COMPLETE)
        assert result is None
        assert sm.state == State.IDLE

    def test_listening_ignores_tts_complete(self):
        sm = StateMachine()
        sm.state = State.LISTENING
        result = sm.transition(AppEvent.TTS_COMPLETE)
        assert result is None
        assert sm.state == State.LISTENING

    def test_idle_ignores_interrupt(self):
        sm = StateMachine()
        result = sm.transition(AppEvent.INTERRUPT)
        assert result is None
        assert sm.state == State.IDLE
