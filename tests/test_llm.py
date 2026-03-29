from types import SimpleNamespace


from voice_app.llm import LLMClient, split_into_sentences


class TestSplitIntoSentences:
    def test_single_sentence_no_split(self):
        result = split_into_sentences("Hello there.")
        assert result == ["Hello there."]

    def test_two_sentences_split_correctly(self):
        result = split_into_sentences("Hello there. How are you?")
        assert result == ["Hello there.", "How are you?"]

    def test_exclamation_and_question(self):
        result = split_into_sentences("That's great! Really? Yes.")
        assert result == ["That's great!", "Really?", "Yes."]

    def test_empty_string_returns_empty_list(self):
        result = split_into_sentences("")
        assert result == []

    def test_no_punctuation_returns_one_item(self):
        result = split_into_sentences("Hey how are you doing today")
        assert result == ["Hey how are you doing today"]

    def test_strips_whitespace(self):
        result = split_into_sentences("  Hello.  World.  ")
        assert len(result) == 2
        assert result[0].strip() == "Hello."


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.finished = []

    def speak(self, text, gen_id):
        self.spoken.append((text, gen_id))

    def finish(self, gen_id):
        self.finished.append(gen_id)


def test_stream_response_emits_sentence_chunks_and_completion():
    import queue
    from unittest.mock import patch

    fake_stream = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello there. "))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="How are you?"))]
        ),
    ]
    fake_tts = FakeTTS()
    event_queue = queue.Queue()
    config = SimpleNamespace(model="gpt-4o-mini", temperature=0.8, max_tokens=500)

    with patch("voice_app.llm.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = fake_stream
        client = LLMClient(
            api_key="test",
            config=config,
            event_queue=event_queue,
            tts_engine=fake_tts,
        )
        client._current_gen_id = 1
        client._stream_response([{"role": "user", "content": "Hi"}], gen_id=1)

    assert fake_tts.spoken == [("Hello there.", 1), ("How are you?", 1)]
    assert fake_tts.finished == [1]
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    assert events[0]["type"] == "FIRST_TTS_CHUNK"
    chunk_events = [event for event in events if event["type"] == "ASSISTANT_RESPONSE_CHUNK"]
    assert [event["text"] for event in chunk_events] == ["Hello there.", "How are you?"]

    completion_event = next(event for event in events if event["type"] == "LLM_RESPONSE_READY")
    assert completion_event["response"] == "Hello there. How are you?"


def test_cancel_invalidates_generation_and_drains_pending_submissions():
    import queue
    from unittest.mock import patch

    with patch("voice_app.llm.OpenAI"):
        client = LLMClient(
            api_key="test",
            config=SimpleNamespace(model="gpt-4o-mini", temperature=0.8, max_tokens=500),
            event_queue=queue.Queue(),
            tts_engine=FakeTTS(),
        )

    client.submit([{"role": "user", "content": "One"}], gen_id=1)
    client.submit([{"role": "user", "content": "Two"}], gen_id=2)
    client.cancel()

    assert client._current_gen_id == 3
    assert client._input_queue.empty()
