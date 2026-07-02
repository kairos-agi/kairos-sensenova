import os
import pickle
import sys
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

_DEFAULT_CFG_PATH = os.environ.get('WAM_DEFAULT_CFG_PATH')
CFG_PATH = os.environ.get('WAM_CFG_PATH', _DEFAULT_CFG_PATH or '')

app = FastAPI(title='WAM Inference Service', version='1.0.0')
engine = None
engine_lock = threading.Lock()


def _get_engine() -> Any:
    global engine
    if not CFG_PATH:
        raise RuntimeError('WAM_CFG_PATH or WAM_DEFAULT_CFG_PATH must be set.')
    if engine is None:
        with engine_lock:
            if engine is None:
                from wam_infer_engine import WamInferEngine

                engine = WamInferEngine(cfg_path=CFG_PATH)
    return engine


@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'cfg_path': CFG_PATH, 'model_loaded': engine is not None}


@app.get('/load_engine')
def load_engine() -> dict:
    _get_engine()
    return {'status': 'ok', 'cfg_path': CFG_PATH, 'model_loaded': engine is not None}


@app.post('/infer')
async def infer(request: Request) -> Response:
    try:
        data = pickle.loads(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'invalid pickle payload: {exc}') from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail='pickle payload must be a dict.')

    try:
        engine_obj = _get_engine()
        output = await run_in_threadpool(lambda: engine_obj.infer(**data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'inference failed: {exc}') from exc

    result = {'success': True, 'output': output}
    return Response(content=pickle.dumps(result), media_type='application/octet-stream')
