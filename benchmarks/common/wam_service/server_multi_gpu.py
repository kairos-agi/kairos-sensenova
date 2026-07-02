import atexit
import os
import pickle
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

_DEFAULT_CFG_PATH = os.environ.get('WAM_DEFAULT_CFG_PATH')
CFG_PATH = os.environ.get('WAM_CFG_PATH', _DEFAULT_CFG_PATH or '')
GPU_IDS = os.environ.get('WAM_GPU_IDS', '0')
REQUEST_TIMEOUT_SEC = int(os.environ.get('WAM_REQUEST_TIMEOUT_SEC', '3600'))
WORKER_STARTUP_TIMEOUT_SEC = int(os.environ.get('WAM_WORKER_STARTUP_TIMEOUT_SEC', '600'))
VERBOSE_LOG = os.environ.get('WAM_VERBOSE_LOG', '1') == '1'


def _parse_gpu_ids(gpu_ids_str: str) -> List[str]:
    gpu_ids = [x.strip() for x in gpu_ids_str.split(',') if x.strip()]
    if not gpu_ids:
        raise ValueError("WAM_GPU_IDS is empty, expected like '0,1,2,3'")
    return gpu_ids


def _worker_main(gpu_id: str, cfg_path: str, in_q: Any, out_q: Any) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    _kairos_wam_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if _kairos_wam_root not in sys.path:
        sys.path.insert(0, _kairos_wam_root)

    try:
        from wam_infer_engine import WamInferEngine

        engine = WamInferEngine(cfg_path=cfg_path)
    except Exception as exc:
        out_q.put(
            {
                'req_id': '__startup__',
                'ok': False,
                'output': None,
                'error': f'worker(gpu={gpu_id}) startup failed: {exc}\n{traceback.format_exc()}',
            }
        )
        return

    out_q.put(
        {
            'req_id': '__startup__',
            'ok': True,
            'output': {'gpu_id': gpu_id},
            'error': None,
        }
    )

    while True:
        item = in_q.get()
        if item is None:
            break

        req_id = item['req_id']
        payload = item['payload']
        try:
            output = engine.infer(**payload)
            out_q.put({'req_id': req_id, 'ok': True, 'output': output, 'error': None})
        except Exception:
            err = traceback.format_exc()
            print(f'[wam_service][worker-error] gpu={gpu_id} req_id={req_id}\n{err}', flush=True)
            out_q.put(
                {
                    'req_id': req_id,
                    'ok': False,
                    'output': None,
                    'error': err,
                }
            )


@dataclass
class _PendingResult:
    event: threading.Event
    ok: Optional[bool] = None
    output: Any = None
    error: Optional[str] = None
    worker_idx: Optional[int] = None
    submit_ts: float = 0.0
    dispatch_ts: float = 0.0


class MultiGpuEnginePool:
    def __init__(self, cfg_path: str, gpu_ids: List[str]):
        self.cfg_path = cfg_path
        self.gpu_ids = gpu_ids
        self.ctx = get_context('spawn')

        self._in_queues: List[Any] = []
        self._out_queue: Any = self.ctx.Queue()
        self._workers: List[Any] = []

        self._schedule_lock = threading.Lock()
        self._worker_inflight: List[int] = []
        self._schedule_rr_cursor = 0

        self._pending: Dict[str, _PendingResult] = {}
        self._pending_lock = threading.Lock()

        self._collector_stop = threading.Event()
        self._collector_thread = threading.Thread(target=self._collector_loop, daemon=True)

        self._start_workers()
        self._collector_thread.start()
        atexit.register(self.shutdown)

    def _start_workers(self) -> None:
        startup_msgs = []
        for gpu_id in self.gpu_ids:
            in_q = self.ctx.Queue()
            proc = self.ctx.Process(
                target=_worker_main,
                args=(gpu_id, self.cfg_path, in_q, self._out_queue),
                daemon=True,
            )
            proc.start()
            self._in_queues.append(in_q)
            self._workers.append(proc)
            self._worker_inflight.append(0)

        for worker_idx, proc in enumerate(self._workers):
            try:
                startup_msgs.append(self._out_queue.get(timeout=WORKER_STARTUP_TIMEOUT_SEC))
            except Exception as exc:
                if not proc.is_alive():
                    raise RuntimeError(
                        f'Worker {worker_idx} (gpu={self.gpu_ids[worker_idx]}) exited before startup '
                        f'(code={proc.exitcode})'
                    ) from exc
                raise RuntimeError(
                    f'Worker {worker_idx} (gpu={self.gpu_ids[worker_idx]}) startup timed out after '
                    f'{WORKER_STARTUP_TIMEOUT_SEC}s'
                ) from exc

        failed = [m for m in startup_msgs if not m.get('ok')]
        if failed:
            raise RuntimeError(f"Worker startup failed: {failed[0].get('error')}")

        print(f'[wam_service] Multi-GPU workers ready on GPUs: {self.gpu_ids}')

    def _pick_worker_idx(self) -> int:
        min_inflight = min(self._worker_inflight)
        candidates = [
            i for i, count in enumerate(self._worker_inflight) if count == min_inflight
        ]
        if not candidates:
            return 0
        pick_at = self._schedule_rr_cursor % len(candidates)
        q_idx = candidates[pick_at]
        self._schedule_rr_cursor = (self._schedule_rr_cursor + 1) % len(self.gpu_ids)
        return q_idx

    def _collector_loop(self) -> None:
        while not self._collector_stop.is_set():
            try:
                msg = self._out_queue.get(timeout=1)
            except Exception:
                continue

            req_id = msg.get('req_id')
            if req_id == '__startup__':
                continue

            with self._pending_lock:
                pending = self._pending.get(req_id)

            if pending is None:
                continue

            pending.ok = msg.get('ok')
            pending.output = msg.get('output')
            pending.error = msg.get('error')
            pending.event.set()

            worker_idx = pending.worker_idx
            if worker_idx is not None:
                with self._schedule_lock:
                    self._worker_inflight[worker_idx] = max(0, self._worker_inflight[worker_idx] - 1)

                if VERBOSE_LOG:
                    done_ts = time.monotonic()
                    queue_wait_ms = int((pending.dispatch_ts - pending.submit_ts) * 1000)
                    infer_ms = int((done_ts - pending.dispatch_ts) * 1000)
                    total_ms = int((done_ts - pending.submit_ts) * 1000)
                    status = 'ok' if pending.ok else 'fail'
                    print(
                        f'[wam_service][done] req_id={req_id} gpu={self.gpu_ids[worker_idx]} '
                        f'status={status} queue_ms={queue_wait_ms} infer_ms={infer_ms} total_ms={total_ms}'
                    )

    def infer(self, payload: Dict[str, Any], timeout_sec: int = REQUEST_TIMEOUT_SEC) -> Any:
        req_id = str(uuid.uuid4())
        now_ts = time.monotonic()
        pending = _PendingResult(event=threading.Event(), submit_ts=now_ts)

        with self._pending_lock:
            self._pending[req_id] = pending

        try:
            with self._schedule_lock:
                q_idx = self._pick_worker_idx()
                self._worker_inflight[q_idx] += 1
                pending.worker_idx = q_idx
                pending.dispatch_ts = time.monotonic()

            if VERBOSE_LOG:
                q_ms = int((pending.dispatch_ts - pending.submit_ts) * 1000)
                print(
                    f'[wam_service][dispatch] req_id={req_id} gpu={self.gpu_ids[q_idx]} '
                    f'inflight={self._worker_inflight[q_idx]} queue_ms={q_ms}'
                )

            self._in_queues[q_idx].put({'req_id': req_id, 'payload': payload})

            done = pending.event.wait(timeout=timeout_sec)
            if not done:
                raise TimeoutError(f'inference timeout after {timeout_sec}s')

            if not pending.ok:
                raise RuntimeError(pending.error or 'unknown worker error')

            return pending.output
        finally:
            if not pending.event.is_set() and pending.worker_idx is not None:
                with self._schedule_lock:
                    self._worker_inflight[pending.worker_idx] = max(
                        0, self._worker_inflight[pending.worker_idx] - 1
                    )
            with self._pending_lock:
                self._pending.pop(req_id, None)

    def shutdown(self) -> None:
        self._collector_stop.set()
        for q in self._in_queues:
            try:
                q.put(None)
            except Exception:
                pass

        for p in self._workers:
            try:
                p.join(timeout=3)
            except Exception:
                pass
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass


app = FastAPI(title='WAM Multi-GPU Inference Service', version='1.0.0')
pool: Optional[MultiGpuEnginePool] = None
pool_lock = threading.Lock()
_pool_load_error: Optional[str] = None
_pool_load_failed: bool = False
_eager_load_thread: Optional[threading.Thread] = None


def _is_pool_loading() -> bool:
    if pool is not None:
        return False
    t = _eager_load_thread
    return t is not None and t.is_alive()


def _get_pool() -> MultiGpuEnginePool:
    global pool
    if not CFG_PATH:
        raise RuntimeError('WAM_CFG_PATH or WAM_DEFAULT_CFG_PATH must be set.')
    if pool is None:
        with pool_lock:
            if pool is None:
                print(
                    f'[wam_service] Loading worker pool on GPUs {_parse_gpu_ids(GPU_IDS)} ...',
                    flush=True,
                )
                pool = MultiGpuEnginePool(cfg_path=CFG_PATH, gpu_ids=_parse_gpu_ids(GPU_IDS))
    return pool


def _background_eager_load() -> None:
    global _pool_load_error, _pool_load_failed
    try:
        _get_pool()
        print('[wam_service] Engine pool eager-loaded on startup', flush=True)
    except Exception as exc:
        _pool_load_failed = True
        err_msg = str(exc).strip() or f'{type(exc).__name__}: worker pool failed to load'
        _pool_load_error = err_msg
        print(f'[wam_service] WARNING: eager load on startup failed: {err_msg}', flush=True)
        traceback.print_exc()


@app.get('/health')
def health() -> dict:
    loaded = pool is not None
    loading = _is_pool_loading()
    return {
        'status': 'ok',
        'cfg_path': CFG_PATH,
        'gpu_ids': _parse_gpu_ids(GPU_IDS),
        'workers_loaded': loaded,
        'workers_loading': loading and not loaded,
        'load_failed': _pool_load_failed,
        'load_error': _pool_load_error,
    }


@app.get('/load_engine')
def load_engine() -> dict:
    global pool
    if pool is not None:
        return {
            'status': 'ok',
            'message': 'engine pool already loaded',
            'already_loaded': True,
            'cfg_path': CFG_PATH,
            'gpu_ids': _parse_gpu_ids(GPU_IDS),
            'workers_loaded': True,
        }

    try:
        _ = _get_pool()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'load_engine failed: {exc}') from exc

    return {
        'status': 'ok',
        'message': 'engine pool loaded',
        'already_loaded': False,
        'cfg_path': CFG_PATH,
        'gpu_ids': _parse_gpu_ids(GPU_IDS),
        'workers_loaded': True,
    }


@app.post('/infer')
async def infer(request: Request) -> Response:
    try:
        data = pickle.loads(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'invalid pickle payload: {exc}') from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail='pickle payload must be a dict.')

    try:
        output = await run_in_threadpool(lambda: _get_pool().infer(data))
    except Exception as exc:
        print(f'[wam_service][infer-error] {exc}', flush=True)
        raise HTTPException(status_code=500, detail=f'inference failed: {exc}') from exc

    result = {'success': True, 'output': output}
    return Response(content=pickle.dumps(result), media_type='application/octet-stream')


@app.on_event('startup')
def on_startup() -> None:
    global _eager_load_thread
    if os.environ.get('WAM_EAGER_LOAD_ON_STARTUP', '1') != '1':
        print('[wam_service] WAM_EAGER_LOAD_ON_STARTUP=0: defer pool load until first /load_engine')
        return
    print('[wam_service] Starting background eager load (HTTP /health available now)', flush=True)
    _eager_load_thread = threading.Thread(target=_background_eager_load, daemon=True)
    _eager_load_thread.start()


@app.on_event('shutdown')
def on_shutdown() -> None:
    global pool
    if pool is not None:
        pool.shutdown()
        pool = None
