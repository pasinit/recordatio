"""Briefing event API routes."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, Request, Response

from tpasini.recordatio.proxy import briefing_events
from tpasini.recordatio.proxy.briefing_log import BriefingLog
from tpasini.recordatio.storage.base import SQLiteStore


@dataclass(frozen=True)
class GetBriefingEventRoute:
    get_store: Callable[[], SQLiteStore]
    get_briefing_log: Callable[[], BriefingLog]
    logger: logging.Logger

    async def __call__(self, request_id: str) -> Response:
        return await get_briefing_event(
            request_id,
            get_store=self.get_store,
            get_briefing_log=self.get_briefing_log,
            logger=self.logger,
        )


@dataclass(frozen=True)
class ListBriefingEventsRoute:
    get_store: Callable[[], SQLiteStore]
    get_briefing_log: Callable[[], BriefingLog]
    logger: logging.Logger

    async def __call__(self, request: Request) -> Response:
        return await list_briefing_events(
            request,
            get_store=self.get_store,
            get_briefing_log=self.get_briefing_log,
            logger=self.logger,
        )


async def get_briefing_event(
    request_id: str,
    *,
    get_store: Callable[[], SQLiteStore],
    get_briefing_log: Callable[[], BriefingLog],
    logger: logging.Logger,
) -> Response:
    return briefing_events.get_briefing_event(request_id, get_briefing_log(), get_store(), logger)


async def list_briefing_events(
    request: Request,
    *,
    get_store: Callable[[], SQLiteStore],
    get_briefing_log: Callable[[], BriefingLog],
    logger: logging.Logger,
) -> Response:
    return briefing_events.list_briefing_events(request, get_briefing_log(), get_store(), logger)


def register_briefing_event_routes(
    app: FastAPI,
    *,
    get_store: Callable[[], SQLiteStore],
    get_briefing_log: Callable[[], BriefingLog],
    logger: logging.Logger,
) -> None:
    get_event_handler = GetBriefingEventRoute(get_store, get_briefing_log, logger)
    list_events_handler = ListBriefingEventsRoute(get_store, get_briefing_log, logger)

    app.add_api_route("/v1/briefing-events/{request_id}", get_event_handler, methods=["GET"])
    app.add_api_route("/v1/briefing-events", list_events_handler, methods=["GET"])
