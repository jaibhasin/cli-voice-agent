from voice_app.echo_guard import EchoTranscriptGuard


def test_matching_transcript_is_treated_as_echo_while_speaking():
    guard = EchoTranscriptGuard(cooldown_ms=2000, interrupted_cooldown_ms=600)
    guard.start_generation(1)
    guard.note_tts_chunk(1, "The weather in Amsterdam is sunny today.")

    assert guard.is_probable_echo(
        "weather in amsterdam is sunny today",
        speaking_active=True,
    )


def test_unrelated_transcript_is_not_treated_as_echo():
    guard = EchoTranscriptGuard(cooldown_ms=2000, interrupted_cooldown_ms=600)
    guard.start_generation(1)
    guard.note_tts_chunk(1, "The weather in Amsterdam is sunny today.")

    assert not guard.is_probable_echo(
        "stop talking and listen",
        speaking_active=True,
    )


def test_recent_assistant_text_is_only_used_within_cooldown():
    guard = EchoTranscriptGuard(cooldown_ms=2000, interrupted_cooldown_ms=600)
    guard.start_generation(1)
    guard.note_tts_chunk(1, "I can help you book a table for tonight.")

    base_ns = 10_000_000_000
    guard.note_tts_complete(1, now_ns=base_ns)

    assert guard.is_probable_echo(
        "help you book a table for tonight",
        speaking_active=False,
        now_ns=base_ns + 500_000_000,
    )
    assert not guard.is_probable_echo(
        "help you book a table for tonight",
        speaking_active=False,
        now_ns=base_ns + 3_000_000_000,
    )
