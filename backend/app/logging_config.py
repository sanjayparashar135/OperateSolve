"""Logging setup that records operations metadata and scrubs secrets."""

from __future__ import annotations

import logging
import traceback

_original_factory = logging.getLogRecordFactory()
_secrets: list[str] = []
_installed = False

QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "openai",
    "openai._base_client",
)


class RedactFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self.secrets = secrets

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.secrets:
            return True
        message = record.getMessage()
        clean = _redact(message, self.secrets)
        if clean != message:
            record.msg = clean
            record.args = ()
        if record.exc_info and record.exc_info[1] is not None:
            text = "".join(traceback.format_exception(*record.exc_info))
            redacted = _redact(text, self.secrets)
            if redacted != text:
                record.exc_info = None
                record.exc_text = redacted
        return True


def install_redaction(secrets: list[str]) -> None:
    """Replace configured secrets in every log record, including tracebacks."""

    global _secrets, _installed
    _secrets = [secret for secret in secrets if len(secret) >= 12]
    if _installed:
        return
    _installed = True

    def factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        record = _original_factory(*args, **kwargs)
        if _secrets:
            message = record.getMessage()
            clean = _redact(message, _secrets)
            if clean != message:
                record.msg = clean
                record.args = ()
            if record.exc_info and record.exc_info[1] is not None:
                text = "".join(traceback.format_exception(*record.exc_info))
                redacted = _redact(text, _secrets)
                if redacted != text:
                    record.exc_info = None
                    record.exc_text = redacted
        return record

    logging.setLogRecordFactory(factory)


def configure_logging(level: str, secrets: list[str] | None = None) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    root.setLevel(numeric_level)
    install_redaction(secrets or [])
    redact = RedactFilter(_secrets)
    for handler in root.handlers:
        handler.filters = [item for item in handler.filters if not isinstance(item, RedactFilter)]
        handler.addFilter(redact)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text
