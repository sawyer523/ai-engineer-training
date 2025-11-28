try:
    from work import config
except Exception:
    import config

def test_session_trim_to_five():
    tid = "trim-5"
    for i in range(10):
        config.append_session_message(tid, "user", f"m{i}")
    msgs = config.get_session_messages(tid, maxlen=5)
    assert len(msgs) == 5
    assert msgs[0]["content"].endswith("5")