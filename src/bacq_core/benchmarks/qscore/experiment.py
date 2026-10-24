"""Run QScore on the myQLM PyLinalg emulator."""

from tempfile import TemporaryDirectory
from pathlib import Path

from qat.pylinalg import PyLinalg
from qat.vsolve.optimize import ScipyMinimizePlugin
from qat.qscore.benchmark import QScore

from .helpers_protocol import check_parameters_value


_MYQLM_EMULATOR = "PyLinalg"


def run(device, **params):
    """Run the QScore core benchmark on the local myQLM emulator."""
    if device != _MYQLM_EMULATOR:
        raise NotImplementedError(
            f"QScore integration currently supports only {_MYQLM_EMULATOR!r}, got {device!r}."
        )

    initial_size = params["initial_size"]
    size_limit = params["size_limit"]
    beta = params["beta"]
    iterator = params["iterator"]
    depth = params["depth"]
    seed = params["seed"]
    check_parameters_value(initial_size, size_limit, beta, iterator, depth, seed)

    qpu = ScipyMinimizePlugin(
        method="COBYLA",
        tol=1e-4,
        options={"maxiter": 300},
    ) | PyLinalg()

    with TemporaryDirectory(prefix="bacq-qscore-") as output_dir:
        output_path = Path(output_dir)
        report = QScore(
            qpu,
            initial_size=initial_size,
            size_limit=size_limit,
            beta=beta,
            iterator=iterator,
            depth=depth,
            output=str(output_path / "qscore.csv"),
            rawdata=str(output_path / "qscore.raw"),
            seed=seed,
        ).run()

    required_report_fields = {"status", "value", "beta", "evaluations", "raw_data"}
    if not isinstance(report, dict) or not required_report_fields.issubset(report):
        raise RuntimeError(
            "The installed qat-qscore version does not expose structured run results. "
            "Install the updated myqlm-compatibility-updates branch."
        )

    return {
        "filetype": "experiment",
        "benchmark": "qscore",
        "device": device,
        "parameters": {
            "initial_size": initial_size,
            "size_limit": size_limit,
            "beta": float(beta),
            "iterator": iterator,
            "depth": depth,
            "seed": seed,
        },
        "result": report,
    }
