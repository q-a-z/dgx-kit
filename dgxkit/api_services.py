"""Services that run beside the models and aren't models: for now Laya, the decision model server."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .laya import LayaService, Problems

router = APIRouter(prefix="/api/services")


def laya(request: Request) -> LayaService:
    s = request.app.state.services
    if s.laya is None:
        s.laya = LayaService(s.state_dir, s.models_root, s.images)
    return s.laya


@router.get("")
async def services(request: Request):
    return [await asyncio.to_thread(laya(request).status)]


@router.post("/laya/start")
async def start(request: Request):
    s, svc = request.app.state.services, laya(request)
    try:
        out = await asyncio.to_thread(svc.start)
    except Problems as e:
        raise HTTPException(409, {"problems": e.problems})
    s.log("start", "laya" + (" (building its image first)" if "preparing" in out else ""))
    return JSONResponse(out, status_code=202 if "preparing" in out else 200)


@router.post("/laya/stop")
async def stop(request: Request):
    stopped = await asyncio.to_thread(laya(request).stop)
    request.app.state.services.log("stop", "laya")
    return {"stopped": stopped}


@router.post("/laya/restart")
async def restart(request: Request):
    svc = laya(request)
    await asyncio.to_thread(svc.stop)
    try:
        out = await asyncio.to_thread(svc.start)
    except Problems as e:
        raise HTTPException(409, {"problems": e.problems})
    request.app.state.services.log("restart", "laya")
    return JSONResponse(out, status_code=202 if "preparing" in out else 200)


@router.put("/laya")
async def configure(body: dict, request: Request):
    svc = laya(request)
    cfg = await asyncio.to_thread(svc.set_config, body.get("device"), body.get("port"), body.get("dir"), body.get("checkpoints"))
    request.app.state.services.log("edit", "laya settings")
    return cfg


@router.get("/laya/logs")
async def logs(request: Request, tail: int = 200):
    return {"logs": await asyncio.to_thread(laya(request).logs, min(tail, 5000))}


@router.get("/laya/key")
async def key(request: Request):
    return {"key": await asyncio.to_thread(laya(request).key)}


@router.post("/laya/test")
async def test(request: Request):
    try:
        return await asyncio.to_thread(laya(request).test)
    except ValueError as e:
        raise HTTPException(502, str(e))
