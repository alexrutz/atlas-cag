"""Start flags: they override environment variables, which override the .env file."""

import os

import pytest

from atlas import __version__
from atlas.__main__ import build_settings, parse_args


@pytest.fixture
def clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in [k for k in os.environ if k.startswith("ATLAS_")]:
        monkeypatch.delenv(k)
    return tmp_path


def settings(*argv):
    return build_settings(parse_args(list(argv)))


def test_flags_override_environment_and_env_file(clean, monkeypatch):
    env = clean / "site.env"
    env.write_text("ATLAS_PORT=9000\nATLAS_HOST=127.0.0.1\nATLAS_LLAMA_SERVER_BIN=/opt/llama-server\n")
    assert settings("--env-file", str(env)).port == 9000
    monkeypatch.setenv("ATLAS_PORT", "9100")
    assert settings("--env-file", str(env)).port == 9100
    s = settings("--env-file", str(env), "--port", "9200", "--host", "0.0.0.0")
    assert (s.port, s.host, s.managed) == (9200, "0.0.0.0", True)
    # --llama-url switches to an external llama-server even if the file names a binary
    s = settings("--env-file", str(env), "--llama-url", "http://gpu-box:8081")
    assert not s.managed and s.llama_url == "http://gpu-box:8081"


def test_data_dir_takes_the_folders_inside_it_along(clean):
    env = clean / ".env"
    env.write_text("ATLAS_DATA_DIR=./data\nATLAS_KV_DIR=./data/kv\nATLAS_MODELS_DIRS=./data/models,/srv/shared-models\n")
    s = settings("--data-dir", "/mnt/big/atlas")
    assert str(s.data_dir) == "/mnt/big/atlas" and str(s.kv_dir) == "/mnt/big/atlas/kv"
    assert s.models_dirs == "/mnt/big/atlas/models,/srv/shared-models"  # a folder elsewhere stays
    s = settings("--data-dir", "/mnt/big/atlas", "--kv-dir", "/fast/kv", "--models-dir", "/m1,/m2")
    assert str(s.kv_dir) == "/fast/kv" and s.models_dirs == "/m1,/m2"
    assert settings().port == 8000 and str(settings().kv_dir) == "data/kv"  # nothing given: unchanged


def test_bad_flags_are_rejected(clean, capsys):
    with pytest.raises(SystemExit) as e:
        parse_args(["--port", "70000"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        parse_args(["--llama-url", "http://x", "--llama-server-bin", "llama-server"])
    assert e.value.code == 2  # one mode or the other
    with pytest.raises(SystemExit, match="settings file not found"):
        settings("--env-file", "missing.env")
    with pytest.raises(SystemExit) as e:
        parse_args(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out


def test_llama_server_port_flags_are_for_the_server_atlas_starts(clean):
    env = clean / "site.env"
    env.write_text("ATLAS_LLAMA_SERVER_BIN=/opt/llama-server\n")
    s = settings("--env-file", str(env), "--llama-port", "8050", "--llama-host", "0.0.0.0")
    assert (s.llama_port, s.llama_host, s.managed) == (8050, "0.0.0.0", True)
    assert s._from_flags == {"llama_port", "llama_host"}  # they win over the address saved in Settings
    assert settings("--env-file", str(env))._from_flags == set()
    # without a llama-server to start, a port makes no sense: say so instead of connecting somewhere
    with pytest.raises(SystemExit, match="--llama-url"):
        settings("--llama-port", "8050")
    with pytest.raises(SystemExit, match="no llama-server is configured"):
        settings("--env-file", str(env), "--llama-url", "http://127.0.0.1:8080", "--llama-port", "8050")
