import logging
from logging.handlers import RotatingFileHandler

import uvicorn

from .config import get_settings

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # persistent log next to the data, so problems can be investigated after a restart
    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "atlas.log", maxBytes=10 * 2**20, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    logging.getLogger("uvicorn.error").addHandler(handler)
    uvicorn.run("atlas.api:create_app", factory=True, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
