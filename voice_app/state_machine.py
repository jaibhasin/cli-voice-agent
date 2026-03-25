from enum import Enum, auto
from typing import Optional


class State(Enum):
    """The high-level stages of a voice conversation."""

    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class AppEvent(Enum):
    """Events emitted by worker threads and consumed by the orchestrator."""

    SPEECH_DETECTED = auto()
    UTTERANCE_COMPLETE = auto()
    FIRST_TTS_CHUNK = auto()
    TTS_COMPLETE = auto()
    INTERRUPT = auto()
    ERROR = auto()
    SHUTDOWN = auto()


TRANSITIONS: dict[tuple[State, AppEvent], State] = {
    (State.IDLE, AppEvent.SPEECH_DETECTED): State.LISTENING,
    (State.LISTENING, AppEvent.UTTERANCE_COMPLETE): State.PROCESSING,
    (State.PROCESSING, AppEvent.FIRST_TTS_CHUNK): State.SPEAKING,
    (State.SPEAKING, AppEvent.TTS_COMPLETE): State.IDLE,
    (State.SPEAKING, AppEvent.INTERRUPT): State.LISTENING,
    (State.PROCESSING, AppEvent.INTERRUPT): State.LISTENING,
}


class StateMachine:
    """Pure transition model for conversation control flow."""

    def __init__(self) -> None:
        self.state: State = State.IDLE

    def transition(self, event: AppEvent) -> Optional[State]:
        """
        Apply a state transition.

        Returns the new state for valid transitions. Returns None for ignored
        transitions and for SHUTDOWN, which acts as an exit signal.
        """

        if event == AppEvent.ERROR:
            self.state = State.IDLE
            return State.IDLE

        if event == AppEvent.SHUTDOWN:
            return None

        next_state = TRANSITIONS.get((self.state, event))
        if next_state is not None:
            self.state = next_state

        return next_state
