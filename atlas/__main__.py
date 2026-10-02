import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from . import __version__
from .config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas: Cache-Augmented Generation over llama.cpp. Flags override environment "
                    "variables, which override the .env file (see .env.example for every setting).")
    parser.add_argument("--version", action="version", version=f"atlas {__version__}")
    web = parser.add_argument_group("web server")
    web.add_argument("--host", help="address to listen on; 0.0.0.0 = reachable from other machines "
                                    "(set ATLAS_API_KEYS then) [ATLAS_HOST, default 127.0.0.1]")
    web.add_argument("--port", type=_port, help="port of the web interface [ATLAS_PORT, default 8000]")
    paths = parser.add_argument_group("files")
    paths.add_argument("--env-file", default=".env", metavar="FILE",
                       help="settings file to read (default: .env in the current directory)")
    paths.add_argument("--data-dir", type=Path, metavar="DIR",
                       help="database, documents and logs; the KV and model folders inside the "
                            "previous data directory move along [ATLAS_DATA_DIR, default ./data]")
    paths.add_argument("--kv-dir", type=Path, metavar="DIR", help="slot files (KV caches) [ATLAS_KV_DIR]")
    paths.add_argument("--models-dir", metavar="DIRS", help="comma-separated folders with GGUF models "
                                                            "[ATLAS_MODELS_DIRS]")
    llama = parser.add_argument_group("llama-server")
    mode = llama.add_mutually_exclusive_group()
    mode.add_argument("--llama-server-bin", metavar="CMD", help="managed mode: start llama-server from "
                                                                "the presets [ATLAS_LLAMA_SERVER_BIN]")
    mode.add_argument("--llama-url", metavar="URL", help="external mode: connect to a running llama-server "
                                                         "(its --slot-save-path must be the KV folder) [ATLAS_LLAMA_URL]")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info",
                        help="log verbosity (default: info)")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    """Settings from flags over environment over the .env file."""
    env_file = args.env_file
    if env_file != ".env" and not Path(env_file).is_file():
        raise SystemExit(f"atlas: settings file not found: {env_file}")
    base = Settings(_env_file=env_file)
    overrides: dict = {}
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.data_dir:
        new = args.data_dir.expanduser()
        overrides["data_dir"] = new
        old = base.data_dir.resolve()

        def moved(path: Path) -> Path:  # a folder inside the previous data directory moves along
            path = path.expanduser().resolve()
            return new / path.relative_to(old) if path.is_relative_to(old) else path

        overrides["kv_dir"] = moved(base.kv_dir)
        overrides["models_dirs"] = ",".join(str(moved(Path(p.strip()))) for p in base.models_dirs.split(",") if p.strip())
    if args.kv_dir:
        overrides["kv_dir"] = args.kv_dir
    if args.models_dir:
        overrides["models_dirs"] = args.models_dir
    if args.llama_server_bin:
        overrides["llama_server_bin"] = args.llama_server_bin
    if args.llama_url:
        overrides.update(llama_url=args.llama_url, llama_server_bin=None)  # external mode
    return Settings(_env_file=env_file, **overrides) if overrides else base


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = build_settings(args)
    level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    # persistent log next to the data, so problems can be investigated after a restart
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "atlas.log", maxBytes=10 * 2**20, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    logging.getLogger("uvicorn.error").addHandler(handler)
    from .api import create_app
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main(sys.argv[1:])
