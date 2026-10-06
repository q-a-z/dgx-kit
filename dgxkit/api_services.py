"""Services that run beside the models and aren't models: the decision model servers (Laya, Lev, Bekko)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .bekko import BekkoService
from .laya import DecisionService, LayaService, Problems
from .lev import LevService

router = APIRouter(prefix="/api/services")

SERVICES: dict[str, type[DecisionService]] = {"laya": LayaService, "lev": LevService, "bekko": BekkoService}


def service(request: Request, name: str) -> DecisionService:
    if name not in SERVICES:
        raise HTTPException(404, "no such service")
    s = request.app.state.services
    if getattr(s, name, None) is None:  # made on first use; tests put fakes here
        setattr(s, name, SERVICES[name](s.state_dir, s.models_root, s.images, downloader=s.downloader))
    return getattr(s, name)


@router.get("")
async def services(request: Request):
    return [await asyncio.to_thread(service(request, n).status) for n in SERVICES]


@router.post("/{name}/download", status_code=202)
async def download(name: str, request: Request):
    """Fetch everything this service needs that isn't on the box (its files, then its image) without starting it."""
    s, svc = request.app.state.services, service(request, name)
    try:
        out = svc.start_download()
    except Problems as e:
        raise HTTPException(409, {"problems": e.problems})
    s.log("download", name)
    return out


@router.post("/{name}/start")
async def start(name: str, request: Request):
    s, svc = request.app.state.services, service(request, name)
    try:
        out = await asyncio.to_thread(svc.start)
    except Problems as e:
        raise HTTPException(409, {"problems": e.problems})
    s.log("start", name + (" (building its image first)" if "preparing" in out else ""))
    return JSONResponse(out, status_code=202 if "preparing" in out else 200)


@router.post("/{name}/stop")
async def stop(name: str, request: Request):
    stopped = await asyncio.to_thread(service(request, name).stop)
    request.app.state.services.log("stop", name)
    return {"stopped": stopped}


@router.post("/{name}/restart")
async def restart(name: str, request: Request):
    svc = service(request, name)
    await asyncio.to_thread(svc.stop)
    try:
        out = await asyncio.to_thread(svc.start)
    except Problems as e:
        raise HTTPException(409, {"problems": e.problems})
    request.app.state.services.log("restart", name)
    return JSONResponse(out, status_code=202 if "preparing" in out else 200)


@router.put("/{name}")
async def configure(name: str, body: dict, request: Request):
    svc = service(request, name)
    cfg = await asyncio.to_thread(svc.set_config, body.get("device"), body.get("port"), body.get("dir"), body.get("checkpoints"), body.get("expose"))
    request.app.state.services.log("edit", f"{name} settings")
    return cfg


@router.get("/{name}/logs")
async def logs(name: str, request: Request, tail: int = 200):
    return {"logs": await asyncio.to_thread(service(request, name).logs, min(tail, 5000))}


@router.get("/{name}/key")
async def key(name: str, request: Request):
    svc = service(request, name)
    if not svc.has_key:
        raise HTTPException(404, f"{svc.title} has no API key")
    return {"key": await asyncio.to_thread(svc.key)}


@router.post("/{name}/test")
async def test(name: str, request: Request):
    try:
        return await asyncio.to_thread(service(request, name).test)
    except ValueError as e:
        raise HTTPException(502, str(e))
