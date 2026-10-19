from fastapi import FastAPI

from tpasini.recordatio import observability


def test_init_tracing_wires_span_and_log_export(monkeypatch):
    calls: list[str] = []

    class FakeProvider:
        def __init__(self, resource):
            self.resource = resource

        def add_span_processor(self, processor):
            calls.append(f"span-processor:{processor!r}")

    class FakeLoggerProvider:
        def __init__(self, resource):
            self.resource = resource

        def add_log_record_processor(self, processor):
            calls.append(f"log-processor:{processor!r}")

    class FakeExporter:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def __repr__(self):
            return f"exporter:{self.endpoint}"

    class FakeFastAPIInstrumentor:
        @staticmethod
        def instrument_app(app, excluded_urls=None):
            calls.append(f"fastapi:{excluded_urls}")

    class FakeInstrumentor:
        def instrument(self, *args, **kwargs):
            calls.append(f"instrument:{kwargs!r}")

    monkeypatch.setattr(observability, "_tracing_enabled", False)
    monkeypatch.setattr(observability.Resource, "create", lambda attrs: attrs)
    monkeypatch.setattr(observability, "TracerProvider", FakeProvider)
    monkeypatch.setattr(observability, "LoggerProvider", FakeLoggerProvider)
    monkeypatch.setattr(observability, "OTLPSpanExporter", FakeExporter)
    monkeypatch.setattr(observability, "OTLPLogExporter", FakeExporter)
    monkeypatch.setattr(observability, "BatchSpanProcessor", lambda exporter: exporter)
    monkeypatch.setattr(observability, "BatchLogRecordProcessor", lambda exporter: exporter)
    monkeypatch.setattr(observability.trace, "set_tracer_provider", lambda provider: calls.append("provider"))
    monkeypatch.setattr(observability._logs, "set_logger_provider", lambda provider: calls.append("logger-provider"))
    monkeypatch.setattr(observability, "FastAPIInstrumentor", FakeFastAPIInstrumentor)
    monkeypatch.setattr(observability, "HTTPXClientInstrumentor", FakeInstrumentor)
    monkeypatch.setattr(observability, "LoggingInstrumentor", FakeInstrumentor)
    monkeypatch.setattr(
        observability,
        "_add_otel_log_handler_once",
        lambda logger_provider: calls.append("root-handler"),
    )

    assert observability.init_tracing(FastAPI(), "http://collector:4317", "recordatio-test") is True

    assert "provider" in calls
    assert "logger-provider" in calls
    assert "root-handler" in calls
    assert "span-processor:exporter:http://collector:4317" in calls
    assert "log-processor:exporter:http://collector:4317" in calls
    assert "fastapi:health" in calls
    assert "instrument:{}" in calls
    assert "instrument:{'set_logging_format': False}" in calls
