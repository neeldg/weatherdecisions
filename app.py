#!/usr/bin/env python3
"""
ASGI entrypoint for PaaS hosts (Railway, Render, etc.) that require app.py / uvicorn.
Serves the static site: index.html at /.
"""
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"


async def homepage(_request):
    if not INDEX.is_file():
        return JSONResponse(
            {"error": "index.html not found in repo root"}, status_code=500
        )
    return FileResponse(INDEX)


async def health(_request):
    return JSONResponse({"status": "ok"})


app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/health", health),
    ]
)
