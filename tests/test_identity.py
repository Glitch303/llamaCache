"""Identity resolution: API key -> frontend, session from header/user field."""

from kv_proxy.identity import resolve_identity, model_of, sanitize

KEYS = {"sillytavern": "sk-st-1", "hermes": "sk-hermes-1"}


def test_key_resolves_frontend():
    idn = resolve_identity({"authorization": "Bearer sk-st-1"}, {}, KEYS)
    assert idn.frontend == "sillytavern"
    assert idn.session == "default"


def test_raw_token_without_bearer():
    idn = resolve_identity({"authorization": "sk-hermes-1"}, {}, KEYS)
    assert idn.frontend == "hermes"


def test_unknown_key_falls_back_to_default():
    idn = resolve_identity({"authorization": "Bearer sk-other"}, {}, KEYS)
    assert idn.frontend == "default"


def test_no_auth_is_default_frontend():
    idn = resolve_identity({}, {"model": "m"}, KEYS)
    assert idn.frontend == "default"


def test_session_from_header_wins():
    idn = resolve_identity(
        {"x-kv-session": "abc123", "authorization": "Bearer sk-st-1"},
        {"user": "from-body"},
        KEYS,
    )
    assert idn.session == "abc123"


def test_session_from_user_field():
    idn = resolve_identity({}, {"user": "hermes-user-42"}, KEYS)
    assert idn.frontend == "default"
    assert idn.session == "hermes-user-42"


def test_session_empty_when_nothing_provided():
    idn = resolve_identity({}, {"model": "m"}, KEYS)
    assert idn.session == "default"


def test_session_from_body_session_field():
    idn = resolve_identity({}, {"session": "chat-77", "model": "m"}, KEYS)
    assert idn.session == "chat-77"


def test_session_from_body_id_field():
    idn = resolve_identity({}, {"id": "chat-99", "model": "m"}, KEYS)
    assert idn.session == "chat-99"


def test_session_from_message_metadata():
    idn = resolve_identity(
        {},
        {
            "model": "m",
            "messages": [
                {"role": "system", "metadata": "chat-42"},
                {"role": "user", "content": "hi"},
            ],
        },
        KEYS,
    )
    assert idn.session == "chat-42"


def test_session_from_message_id():
    idn = resolve_identity(
        {},
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "id": "asst-13"},
            ],
        },
        KEYS,
    )
    assert idn.session == "asst-13"


def test_session_header_beats_body_hints():
    idn = resolve_identity(
        {"x-kv-session": "explicit"},
        {"session": "body-1", "user": "u"},
        KEYS,
    )
    assert idn.session == "explicit"


def test_body_hint_beats_user_field():
    idn = resolve_identity({}, {"session": "s1", "user": "u"}, KEYS)
    assert idn.session == "s1"


def test_non_string_id_and_empty_hints_ignored():
    idn = resolve_identity(
        {},
        {"id": 123, "session": "  ", "model": "m",
         "messages": [{"role": "user", "content": "x", "metadata": ""}]},
        KEYS,
    )
    assert idn.session == "123"


def test_numeric_session_id():
    idn = resolve_identity({}, {"session": 42}, KEYS)
    assert idn.session == "42"


# ------------------------------------------------- content-derived sessions

TURN_1 = {"model": "m", "messages": [{"role": "user", "content": "tell me a joke"}]}
TURN_2 = dict(TURN_1)
TURN_2["messages"] = [
    {"role": "user", "content": "tell me a joke"},
    {"role": "assistant", "content": "why did the llama..."},
    {"role": "user", "content": "another one"},
]


def test_content_session_stable_across_turns():
    a = resolve_identity({}, TURN_1, KEYS).session
    b = resolve_identity({}, TURN_2, KEYS).session
    assert a == b
    assert a.startswith("c-")
    assert a != "default"


def test_content_session_differs_between_chats():
    other = {"model": "m", "messages": [{"role": "user", "content": "write a haiku about rain"}]}
    assert resolve_identity({}, TURN_1, KEYS).session != \
        resolve_identity({}, other, KEYS).session


def test_content_session_with_system_prompt():
    base = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "ahoy"},
    ]
    turn2 = base + [{"role": "assistant", "content": "aye"}, {"role": "user", "content": "more"}]
    assert resolve_identity({}, {"messages": base}, KEYS).session == \
        resolve_identity({}, {"messages": turn2}, KEYS).session


def test_same_system_different_first_user_differs():
    sysm = {"role": "system", "content": "You are a pirate."}
    a = {"messages": [sysm, {"role": "user", "content": "ahoy"}]}
    b = {"messages": [sysm, {"role": "user", "content": "where be the gold"}]}
    assert resolve_identity({}, a, KEYS).session != \
        resolve_identity({}, b, KEYS).session


def test_content_session_beats_user_field():
    idn = resolve_identity(
        {}, {"user": "shyn", "messages": [{"role": "user", "content": "hi"}]}, KEYS
    )
    assert idn.session.startswith("c-")
    assert idn.session != "shyn"


def test_explicit_hints_beat_content_session():
    body = {"session": "pinned", "messages": [{"role": "user", "content": "hi"}]}
    assert resolve_identity({}, body, KEYS).session == "pinned"
    body = {"messages": [{"role": "user", "content": "hi", "metadata": "meta-1"}]}
    assert resolve_identity({}, body, KEYS).session == "meta-1"


def test_content_session_legacy_prompt():
    a = resolve_identity({}, {"prompt": "User: hello\nAssistant: hi\n"}, KEYS).session
    b = resolve_identity({}, {"prompt": "User: hello\nAssistant: hi\nUser: again\n"}, KEYS).session
    assert a == b  # same opening -> same session
    c = resolve_identity({}, {"prompt": "User: different start\n"}, KEYS).session
    assert c != a


def test_no_messages_no_user_still_default():
    idn = resolve_identity({}, {"model": "m"}, KEYS)
    assert idn.session == "default"


def test_sanitize_makes_names_fs_safe():
    assert sanitize("My Session/01") == "My_Session_01" or "-" in sanitize("My Session/01")
    assert sanitize("") == "default"
    assert sanitize("ok-name.1") == "ok-name.1"


def test_sanitize_distinct_names_stay_distinct():
    a = sanitize("a/b")
    b = sanitize("a*b")
    assert a != b


def test_model_of():
    assert model_of({"model": "my-model"}) == "my-model"
    assert model_of({}) == ""
    assert model_of(None) == ""
