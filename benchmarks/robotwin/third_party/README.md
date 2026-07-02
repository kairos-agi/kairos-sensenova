# RoboTwin third-party dependencies

Place the RoboTwin repository here as `RoboTwin/`:

```bash
# Clone RoboTwin into this directory
cd benchmarks/robotwin/third_party
git clone https://github.com/TianxingChen/RoboTwin.git
```

The eval scripts expect `ROBOTWIN_ROOT` to point to the RoboTwin repo root.
By default this is set to `<benchmark_root>/third_party/RoboTwin` in `eval_run.sh`.
You can override it with:

```bash
export ROBOTWIN_ROOT=/path/to/your/RoboTwin
```
