#!/usr/bin/env bash
# Shared EGL setup and preflight helpers for LIBERO evaluation scripts.

nvidia_egl_library_available() {
  "${EVAL_PYTHON:?EVAL_PYTHON must be set}" - <<'PY' >/dev/null 2>&1
import ctypes

ctypes.CDLL("libEGL_nvidia.so.0")
PY
}

libero_configure_egl_backend() {
  local vendor_mode="${LIBERO_EGL_VENDOR_MODE:-nvidia}"
  local vendor_json

  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

  case "${vendor_mode}" in
    nvidia|auto)
      if nvidia_egl_library_available; then
        vendor_json="${LIBERO_NVIDIA_EGL_VENDOR_JSON:-${OUTPUT_ROOT:?OUTPUT_ROOT must be set}/nvidia_egl_vendor.json}"
        mkdir -p "$(dirname "${vendor_json}")"
        cat >"${vendor_json}" <<'EOF'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
EOF
        if [[ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" || "${vendor_mode}" == "nvidia" ]]; then
          export __EGL_VENDOR_LIBRARY_FILENAMES="${vendor_json}"
        fi
        export LIBERO_EGL_VENDOR_MODE_RESOLVED="nvidia"
        export LIBERO_EGL_DEVICE_ID_MODE="${LIBERO_EGL_DEVICE_ID_MODE:-physical}"
        echo "[egl] using NVIDIA EGL vendor: __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES}"
      elif [[ "${vendor_mode}" == "nvidia" ]]; then
        echo "[egl] ERROR: libEGL_nvidia.so.0 is not loadable from EVAL_PYTHON=${EVAL_PYTHON}" >&2
        echo "[egl] Set LIBERO_EGL_VENDOR_MODE=system only if system GLVND EGL is known to enumerate NVIDIA devices." >&2
        return 3
      else
        export LIBERO_EGL_VENDOR_MODE_RESOLVED="system"
        export LIBERO_EGL_DEVICE_ID_MODE="${LIBERO_EGL_DEVICE_ID_MODE:-local}"
        echo "[egl] NVIDIA EGL vendor not loadable; falling back to system GLVND vendor discovery"
      fi
      ;;
    system)
      export LIBERO_EGL_VENDOR_MODE_RESOLVED="system"
      export LIBERO_EGL_DEVICE_ID_MODE="${LIBERO_EGL_DEVICE_ID_MODE:-local}"
      echo "[egl] using system GLVND EGL vendor discovery"
      ;;
    *)
      echo "[egl] ERROR: invalid LIBERO_EGL_VENDOR_MODE=${vendor_mode}; expected nvidia, auto, or system" >&2
      return 3
      ;;
  esac
}

libero_egl_device_count_current() {
  local count
  count="$("${EVAL_PYTHON:?EVAL_PYTHON must be set}" - <<'PY' 2>/dev/null || true
try:
    from mujoco.egl import egl_ext as EGL
    print(len(EGL.eglQueryDevicesEXT()))
except Exception:
    print("ERR")
PY
)"
  count="$(echo "${count}" | tail -n 1 | tr -d '[:space:]')"
  if [[ "${count}" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "${count}"
  else
    printf '0\n'
  fi
}

libero_print_egl_diagnostics() {
  echo "[egl] diagnostics:"
  echo "[egl]   MUJOCO_GL=${MUJOCO_GL:-} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-}"
  echo "[egl]   __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-}"
  echo "[egl]   __EGL_VENDOR_LIBRARY_DIRS=${__EGL_VENDOR_LIBRARY_DIRS:-}"
  echo "[egl]   LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  if [[ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]]; then
    echo "[egl]   /usr/lib/x86_64-linux-gnu/libEGL.so.1 -> $(readlink -f /usr/lib/x86_64-linux-gnu/libEGL.so.1)"
    sha256sum /usr/lib/x86_64-linux-gnu/libEGL.so.1 2>/dev/null | sed 's/^/[egl]   sha256 /' || true
  fi
  if [[ -d /etc/glvnd/egl_vendor.d ]]; then
    echo "[egl]   /etc/glvnd/egl_vendor.d:"
    ls -l /etc/glvnd/egl_vendor.d 2>/dev/null | sed 's/^/[egl]     /' || true
  fi
  if [[ -d /usr/share/glvnd/egl_vendor.d ]]; then
    echo "[egl]   /usr/share/glvnd/egl_vendor.d:"
    ls -l /usr/share/glvnd/egl_vendor.d 2>/dev/null | sed 's/^/[egl]     /' || true
  fi
  if [[ -d /dev/dri ]]; then
    echo "[egl]   /dev/dri:"
    ls -l /dev/dri 2>/dev/null | sed 's/^/[egl]     /' || true
  else
    echo "[egl]   /dev/dri: missing"
  fi
}

libero_preflight_egl_devices() {
  if [[ "${MUJOCO_GL,,}" != "egl" ]]; then
    echo "[egl] MUJOCO_GL=${MUJOCO_GL}; skip EGL device preflight"
    return 0
  fi

  local egl_count
  egl_count="$(libero_egl_device_count_current)"
  if [[ ! "${egl_count}" =~ ^[0-9]+$ ]] || [[ "${egl_count}" -le 0 ]]; then
    echo "[egl] ERROR: eglQueryDevicesEXT returned ${egl_count}; MuJoCo/robosuite EGL rendering will fail." >&2
    echo "[egl] This is an environment problem, not a CUDA_VISIBLE_DEVICES mapping problem." >&2
    libero_print_egl_diagnostics >&2
    return 3
  fi
  echo "[egl] EGL preflight ok: eglQueryDevicesEXT=${egl_count}, device_id_mode=${LIBERO_EGL_DEVICE_ID_MODE:-local}"
}

libero_preflight_worker_render_contexts() {
  local label="$1"
  local gpu_ids="$2"
  local seen=","
  local gpu output

  IFS=',' read -r -a _gpu_id_parts <<<"${gpu_ids}"
  for gpu in "${_gpu_id_parts[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -z "${gpu}" ]] && continue
    if [[ "${seen}" == *",${gpu},"* ]]; then
      continue
    fi
    seen="${seen}${gpu},"

    if ! output="$(env -u NVIDIA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" "${EVAL_PYTHON:?EVAL_PYTHON must be set}" - <<'PY' 2>&1
import os

physical = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
mode = os.environ.get("LIBERO_EGL_DEVICE_ID_MODE", "local").strip().lower()

from mujoco.egl import egl_ext as EGL
egl_count = len(EGL.eglQueryDevicesEXT())
from robosuite.utils.binding_utils import GLContext

egl_device = physical
if mode not in {"physical", "global", "as_is", "none"}:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    egl_device = str(visible.index(physical)) if physical in visible else "0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = egl_device

ctx = GLContext(16, 16, int(egl_device))
ctx.make_current()
ctx.free()
print(f"ok egl_count={egl_count} egl_device={egl_device} mode={mode}")
PY
)"; then
      echo "[${label}] ERROR: EGL render context preflight failed for GPU ${gpu}" >&2
      echo "${output}" >&2
      libero_print_egl_diagnostics >&2
      return 3
    fi
    echo "[${label}] EGL render context preflight ok for GPU ${gpu}: $(echo "${output}" | tail -n 1)"
  done
}
