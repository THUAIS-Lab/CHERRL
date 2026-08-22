#!/usr/bin/env python3
"""
Least-connections async reverse proxy for vLLM backends.
Optimized for concurrent (non-streaming) API calls.
"""
import argparse
import logging
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LB] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lb")

class Backend:
    def __init__(self, url: str):
        self.url = url
        self.active = 0

backends: list[Backend] = []

def pick_backend() -> Backend:
    return min(backends, key=lambda b: b.active)

async def proxy(request: web.Request) -> web.Response:
    backend = pick_backend()
    backend.active += 1
    target = backend.url + str(request.rel_url)
    try:
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        async with request.app["session"].request(
            method=request.method,
            url=target,
            headers=headers,
            data=body,
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                status=resp.status,
                headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("transfer-encoding", "content-encoding")
                },
                body=resp_body,
            )
    except Exception as e:
        log.error("Backend %s error: %s", backend.url, e)
        return web.Response(status=502, text=f"Bad gateway: {e}")
    finally:
        backend.active -= 1

async def health(request: web.Request) -> web.Response:
    info = [{"backend": b.url, "active_requests": b.active} for b in backends]
    return web.json_response({"status": "ok", "backends": info})

async def on_startup(app: web.Application):
    connector = TCPConnector(limit=0, keepalive_timeout=30)
    timeout = ClientTimeout(total=1200, connect=10)
    app["session"] = ClientSession(connector=connector, timeout=timeout)
    log.info("Load balancer ready. Backends: %s", [b.url for b in backends])

async def on_shutdown(app: web.Application):
    await app["session"].close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--lb-port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    for port in args.ports:
        backends.append(Backend(f"http://127.0.0.1:{port}"))

    app = web.Application()
    app.router.add_route("*", "/health_lb", health)
    app.router.add_route("*", "/{path_info:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.lb_port, access_log=None)

if __name__ == "__main__":
    main()
