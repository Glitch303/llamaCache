"""Config tests: upstream API key resolution (explicit / llama-swap config / env)."""

import pytest

from kv_proxy.config import Config, load_config, resolve_upstream_api_key


def test_no_key_configured(tmp_path):
    cfg = Config()
    assert resolve_upstream_api_key(cfg) is None


def test_explicit_key_wins(tmp_path):
    swap_cfg = tmp_path / "llama-swap.yaml"
    swap_cfg.write_text("apiKeys:\n  - ls-from-file\n  - ls-second\n")
    cfg = Config(llama_swap_api_key="ls-explicit", llama_swap_config=swap_cfg)
    assert resolve_upstream_api_key(cfg) == "ls-explicit"


def test_key_from_llama_swap_config(tmp_path):
    swap_cfg = tmp_path / "llama-swap.yaml"
    swap_cfg.write_text(
        "listen: 0.0.0.0:27906\napiKeys:\n  - ls-first\n  - ls-second\n",
    )
    cfg = Config(llama_swap_config=swap_cfg)
    assert resolve_upstream_api_key(cfg) == "ls-first"


def test_llama_swap_config_missing_file(tmp_path):
    cfg = Config(llama_swap_config=tmp_path / "nope.yaml")
    assert resolve_upstream_api_key(cfg) is None


def test_llama_swap_config_bad_yaml(tmp_path):
    swap_cfg = tmp_path / "llama-swap.yaml"
    swap_cfg.write_text("apiKeys: [unclosed\n")
    cfg = Config(llama_swap_config=swap_cfg)
    assert resolve_upstream_api_key(cfg) is None


def test_load_config_reads_key_and_yaml_path(tmp_path, monkeypatch):
    toml = tmp_path / "config.toml"
    swap_cfg = tmp_path / "llama-swap.yaml"
    swap_cfg.write_text("apiKeys:\n  - ls-file\n")
    toml.write_text(
        f'listen = "127.0.0.1:1"\nllama_swap_config = "{swap_cfg}"\n',
    )
    cfg = load_config(toml)
    assert cfg.llama_swap_api_key is None
    assert cfg.llama_swap_config == swap_cfg
    assert resolve_upstream_api_key(cfg) == "ls-file"


def test_load_config_env_override(tmp_path, monkeypatch):
    toml = tmp_path / "config.toml"
    toml.write_text('listen = "127.0.0.1:1"\n')
    monkeypatch.setenv("KV_PROXY_LLAMA_SWAP_API_KEY", "ls-env")
    cfg = load_config(toml)
    assert cfg.llama_swap_api_key == "ls-env"
    assert resolve_upstream_api_key(cfg) == "ls-env"


def test_explicit_key_in_toml(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('listen = "127.0.0.1:1"\nllama_swap_api_key = "ls-toml"\n')
    cfg = load_config(toml)
    assert resolve_upstream_api_key(cfg) == "ls-toml"
