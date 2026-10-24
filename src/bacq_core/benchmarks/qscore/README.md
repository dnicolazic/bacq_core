# BACQ-OP-QSCORE with the myQLM emulator

This BACQ benchmark runs the QScore implementation from the separate
`dnicolazic/qscore` repository. QScore creates the random MaxCut instances,
constructs QAOA jobs, executes them on myQLM's local `PyLinalg` emulator, and
computes the QScore boundary. BACQ starts the run and stores the experiment and
benchmark result using its regular output format.

This integration currently supports only the local `PyLinalg` emulator. It
does not run on the Qiskit IBM devices used by some other BACQ benchmarks.

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

Use a BACQ protocol input with `"benchmark": "qscore"` and `"device":
"PyLinalg"`. For example:

```bash
python src/bacq_core/main.py protocol_qscore.json
```

The supplied `examples/protocol_qscore.json` uses a small graph-size range as
an example. QScore evaluates 100 random graph instances for each tested size,
so even emulator runs can take time. Increase `size_limit` for a larger sweep.

Parameters:

- `initial_size`: smallest MaxCut graph size to test.
- `size_limit`: largest MaxCut graph size to test.
- `beta`: QScore threshold ratio.
- `iterator`: `dichotomic` (binary search) or `exhaustive`.
- `depth`: QAOA depth.
- `seed`: integer graph-generation seed, or `null` for a random seed.

QScore performs the graph-size sweep itself; use a single BACQ `protocol` input,
not `protocol_seq`. BACQ records the QScore boundary or bound as its metric.
`success` is true when QScore identifies an exact boundary between a passing
and a failing size; it is false when the result is only an upper or lower bound.
