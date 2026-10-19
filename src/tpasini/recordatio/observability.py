"""OpenTelemetry tracing setup for Recordatio.

Tracing is opt-in. When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset, setup is a
no-op and spans use OpenTelemetry's proxy tracer without exporting anything.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI
from opentelemetry import _logs, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer

log = logging.getLogger("tpasini.recordatio.observability")

_TRACER_NAME = "tpasini.recordatio"
_tracing_enabled = False


def init_tracing(app: FastAPI, endpoint: str = "", service_name: str = "recordatio") -> bool:
    """Initialise tracing once per process and instrument the FastAPI app."""
    global _tracing_enabled

    endpoint = endpoint.strip()
    service_name = service_name.strip() or "recordatio"

    if _tracing_enabled:
        return True

    if not endpoint:
        log.info("OTel: no OTLP endpoint configured - tracing disabled")
        return False

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(endpoint=endpoint)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    _logs.set_logger_provider(logger_provider)

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=False)
    _add_otel_log_handler_once(logger_provider)

    log.info("OTel: tracing/logging enabled - service=%r exporting to %s", service_name, endpoint)
    _tracing_enabled = True
    return True


def _add_otel_log_handler_once(logger_provider: LoggerProvider) -> None:
    """Attach the OTel log exporter to Python logging once per process."""
    root_logger = logging.getLogger()
    if any(getattr(handler, "_recordatio_otel_log_handler", False) for handler in root_logger.handlers):
        return
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    handler._recordatio_otel_log_handler = True
    root_logger.addHandler(handler)


def tracer() -> Tracer:
    """Return the module-level tracer. Safe before ``init_tracing``."""
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Start a span with primitive attributes and record raised exceptions."""
    with tracer().start_as_current_span(name) as sp:
        for key, value in attrs.items():
            if value is not None:
                sp.set_attribute(key, value)
        try:
            yield sp
        except Exception as exc:
            mark_error(sp, str(exc), exc)
            raise


def mark_error(sp: Span, description: str, exc: BaseException | None = None) -> None:
    """Mark ``sp`` as failed without forcing callers to raise an exception."""
    if exc is not None:
        sp.record_exception(exc)
    sp.set_status(Status(StatusCode.ERROR, description))
