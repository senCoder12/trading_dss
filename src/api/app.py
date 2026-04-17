"""
FastAPI application factory.

Create the app with ``create_app()`` or import the pre-built ``app``
singleton for ASGI deployment.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.logging_config import setup_logging
from config.settings import settings
from src.api.routes import (
    anomalies,
    backtest,
    indices,
    market_data,
    news,
    portfolio,
    signals,
    system,
)
from src.api.websocket import broadcaster, process_event_queue
from src.utils.date_utils import get_ist_now

# Resolved once at import time; works regardless of working directory.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown logic."""
    setup_logging(
        log_dir=settings.logging.log_dir,
        console_level=getattr(logging, settings.logging.level),
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    )
    logger.info("trading_dss API v1.0.0 starting (env=%s)", settings.environment)

    # Start the WebSocket event-queue processor (drains sync→async queue)
    import asyncio
    queue_task = asyncio.create_task(process_event_queue())

    yield

    queue_task.cancel()
    logger.info("trading_dss API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Trading Decision Support System API",
        description="REST API for Indian market trading signals and analysis",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # CORS for local React dev servers
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────
    @application.middleware("http")
    async def log_slow_requests(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        if elapsed > 1.0:
            logger.warning(
                "Slow request: %s %s took %.2fs",
                request.method,
                request.url.path,
                elapsed,
            )
        return response

    # ── Full request/response logging (enable via LOG_HTTP_BODIES=1) ──────
    if os.getenv("LOG_HTTP_BODIES") == "1":
        from starlette.responses import Response as StarletteResponse

        _MAX_BODY_LOG = 4096  # bytes

        def _truncate(data: bytes) -> str:
            text = data.decode("utf-8", errors="replace")
            if len(text) > _MAX_BODY_LOG:
                return text[:_MAX_BODY_LOG] + f"...<truncated {len(text) - _MAX_BODY_LOG} chars>"
            return text

        @application.middleware("http")
        async def log_http_bodies(request: Request, call_next):
            req_body = await request.body()

            async def receive():
                return {"type": "http.request", "body": req_body, "more_body": False}

            request = Request(request.scope, receive)

            logger.info(
                "→ %s %s%s  body=%s",
                request.method,
                request.url.path,
                f"?{request.url.query}" if request.url.query else "",
                _truncate(req_body) if req_body else "<empty>",
            )

            response = await call_next(request)

            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk

            logger.info(
                "← %s %s → %d  body=%s",
                request.method,
                request.url.path,
                response.status_code,
                _truncate(resp_body) if resp_body else "<empty>",
            )

            return StarletteResponse(
                content=resp_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

    # ── Route modules ────────────────────────────────────────────────────
    application.include_router(
        market_data.router, prefix="/api/market", tags=["Market Data"],
    )
    application.include_router(
        signals.router, prefix="/api/signals", tags=["Signals"],
    )
    application.include_router(
        indices.router, prefix="/api/indices", tags=["Indices"],
    )
    application.include_router(
        portfolio.router, prefix="/api/portfolio", tags=["Portfolio"],
    )
    application.include_router(
        news.router, prefix="/api/news", tags=["News"],
    )
    application.include_router(
        anomalies.router, prefix="/api/anomalies", tags=["Anomalies"],
    )
    application.include_router(
        system.router, prefix="/api/system", tags=["System"],
    )
    application.include_router(
        backtest.router, prefix="/api/backtest", tags=["Backtest"],
    )

    # ── Health check ─────────────────────────────────────────────────────
    @application.get("/api/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "timestamp": get_ist_now().isoformat()}

    # ── WebSocket endpoint ────────────────────────────────────────────
    @application.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await broadcaster.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            await broadcaster.disconnect(websocket)
        except Exception:
            await broadcaster.disconnect(websocket)

    # ── Global error handler ─────────────────────────────────────────────
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled API error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ── Production SPA serving ────────────────────────────────────────────────
    # Registered AFTER all /api/* routes so FastAPI's explicit-route matching
    # always wins over the catch-all path parameter.  Only activated when the
    # React build exists (after `cd frontend && npm run build`).
    if _FRONTEND_DIST.exists():
        assets_dir = _FRONTEND_DIST / "assets"
        if assets_dir.exists():
            application.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="static-assets",
            )
            logger.info("Serving frontend assets from %s", assets_dir)

        @application.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            """
            Catch-all that returns ``index.html`` so React Router can handle
            client-side navigation.  All ``/api/*`` paths are matched by their
            own routers before this handler is ever reached.
            """
            index_path = _FRONTEND_DIST / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Frontend not built. Run: cd frontend && npm run build"
                },
            )
    else:
        logger.info(
            "Frontend build not found at %s — skipping SPA serving. "
            "Run `cd frontend && npm run build` to enable.",
            _FRONTEND_DIST,
        )

    return application


app = create_app()
