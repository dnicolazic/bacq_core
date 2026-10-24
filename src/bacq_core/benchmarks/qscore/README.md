# BACQ-OP-QSCORE with myQLM simulators

This BACQ benchmark runs the QScore implementation from the separate
`dnicolazic/qscore` repository on `CLinalg`. It also supports two local
annealing modes, implemented in BACQ with the same graph-size search and QScore
metric: myQLM's `QutipQPU` uses analog quantum optimization (AQO) jobs, and
`SimulatedAnnealing` uses MaxCut annealing jobs. These modes do not execute
QScore's QAOA circuits.

All three devices are local emulators; they do not use BACQ's Qiskit IBM
devices. The QutipQPU path requires the myQLM analog/Qutip simulator
dependencies. SimulatedAnnealing is imported from
`qat.simulated_annealing`.

When using a local `qscore` checkout, do not put the checkout root first on
`PYTHONPATH`: its `qat/__init__.py` shadows myQLM's compiled `qat` package, and
QutipQPU then fails with `ImportError: cannot import name 'get_default_qpu'
from 'qat.qpus'`. Install `qat-qscore` (it only ships `qat.qscore`) or append
`qscore/qat` to the already-imported `qat.__path__`.

## Install

From the `bacq_core` repository root, install the optional dependency:

```bash
python -m pip install -e ".[qscore]"
```

The `qscore` extra installs `qat-qscore` from the
`myqlm-compatibility-updates` branch of
[dnicolazic/qscore](https://github.com/dnicolazic/qscore). This is separate
from BACQ's required dependencies.

## Run

Use a BACQ protocol input with `"benchmark": "qscore"`. Set `"device"` to
`"CLinalg"`, `"QutipQPU"`, or `"SimulatedAnnealing"`. The included example
protocols use small graph sizes:

```bash
python src/bacq_core/main.py examples/protocol_qscore.json
python src/bacq_core/main.py examples/protocol_qscore_qutip.json
python src/bacq_core/main.py examples/protocol_qscore_simulated_annealing.json
```

Each mode evaluates 100 random graph instances per tested size by default, so
even emulator runs can take time. Increase `size_limit` for a larger sweep.
`depth` is used for the QAOA/CLinalg path and ignored by the annealing devices.
The workbench runner also allows reducing instances and changing emulator
steps/shots for fast experiments.

Parameters:

- `initial_size`: smallest MaxCut graph size to test.
- `size_limit`: largest MaxCut graph size to test.
- `beta`: QScore threshold ratio.
- `iterator`: `dichotomic` (binary search) or `exhaustive` (consecutive sizes
  until the first failing size or the inclusive limit).
- `depth`: QAOA depth.
- `seed`: non-negative integer graph-generation seed, or `null` for a random
  seed.
- `qutip_steps`: positive integer evolution discretization for QutipQPU
  (default 100).
- `qutip_tmax`: finite positive AQO evolution time for QutipQPU (default 5.0,
  matching the reference BACQ QutipQPU notebook). Saved with the result
  parameters. Previously it was hard-coded to 1.0, which produced a ratio
  below the threshold at size 5 and stopped the search immediately.
  Other annealers retain their existing evolution time.

QScore performs the graph-size sweep itself; use a single BACQ `protocol` input,
not `protocol_seq`. BACQ records the QScore boundary or bound as its metric.
`success` is true when QScore identifies an exact boundary between a passing
and a failing size; it is false when the result is only an upper or lower bound.
A size passes only when its ratio is strictly greater than `beta`; equality is
a failing size. Thus the exact boundary is the passing size immediately before
the first failing size. `dichotomic` assumes this pass/fail behavior is
monotonic over graph sizes. When `initial_size` equals `size_limit`, the single
size is evaluated once and can only produce an upper or lower bound, not an
exact boundary. The per-instance seeds advance consecutively across all
evaluated sizes and are included in the raw data.

All QAOA and annealing paths save `duration_seconds` per instance in `raw_data`
and `instance_duration_sum` in each size evaluation. Timings measure QPU
submission/result resolution, including the QAOA optimizer where applicable,
but exclude graph/job construction and result-file writing. The workbench uses
these measured values for mean-instance and total-per-size runtime curves.
