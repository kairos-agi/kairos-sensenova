# KAIROS WAM Benchmarks

This directory is the cleaned benchmark export tree. It starts from the known-working
`benchmarks/` layout and only factors out shared benchmark plumbing into `common/`.

## Layout

- `common/adapters/`: external model adapters shared by LIBERO and LIBERO-Plus.
- `common/clients/`: HTTP client utilities shared by benchmark policies.
- `common/wam_service/`: shared WAM FastAPI service and inference engine.
- `common/scripts/`: reusable helper scripts.
- `libero_plus/`, `robotwin/`: benchmark-specific code and launch scripts.

The benchmark directories keep their original entrypoints and import paths. Files such as
`kairoswam.models.external_model_adapter` and `wam_service.server_multi_gpu` are thin
compatibility wrappers that point to the shared implementation in `common/`.

## Notes

- `benchmarks/` was not modified while creating this tree.
- Runtime outputs, temporary folders, and Python cache files are intentionally excluded.
- `WAM_CFG_PATH` still overrides the benchmark default config. If it is not set, each
  benchmark wrapper provides its own default config path.
