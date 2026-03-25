from voice_app.history import append_message, load_history, save_history


def test_load_returns_empty_list_when_file_missing(tmp_path):
    result = load_history(str(tmp_path / "no_such_file.json"))
    assert result == []


def test_save_and_reload_round_trip(tmp_path):
    filepath = str(tmp_path / "history.json")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    save_history(filepath, messages)
    loaded = load_history(filepath)
    assert loaded == messages


def test_append_adds_message_and_persists(tmp_path):
    filepath = str(tmp_path / "history.json")
    save_history(filepath, [{"role": "user", "content": "First"}])

    result = append_message(filepath, "assistant", "Second", max_messages=50)

    assert len(result) == 2
    assert result[1] == {"role": "assistant", "content": "Second"}
    assert len(load_history(filepath)) == 2


def test_append_trims_to_max_messages(tmp_path):
    filepath = str(tmp_path / "history.json")
    initial = [{"role": "user", "content": str(i)} for i in range(5)]
    save_history(filepath, initial)

    result = append_message(filepath, "assistant", "new", max_messages=5)

    assert len(result) == 5
    assert result[0]["content"] == "1"
    assert result[-1]["content"] == "new"


def test_append_creates_file_if_missing(tmp_path):
    filepath = str(tmp_path / "new_history.json")
    result = append_message(filepath, "user", "Hello", max_messages=50)
    assert result == [{"role": "user", "content": "Hello"}]
    assert (tmp_path / "new_history.json").exists()
