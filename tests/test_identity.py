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
