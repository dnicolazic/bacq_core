"""Run QScore using myQLM simulators."""

import math
from tempfile import TemporaryDirectory
from pathlib import Path

from . import annealing
from .helpers_protocol import check_parameters_value


_QAOA_DEVICES = {"CLinalg", "MPS"}
_SUPPORTED_DEVICES = _QAOA_DEVICES | {
    "QutipQPU",
    "AnalogQPU",
    "SQAQPU",
    "SimulatedAnnealing",
    "MPSTraj",
}


def _build_clinalg_qpu():
    from qat.clinalg import CLinalg
    from qat.vsolve.optimize import ScipyMinimizePlugin

    return ScipyMinimizePlugin(
        method="COBYLA",
        tol=1e-4,
        options={"maxiter": 300},
    ) | CLinalg()


def _build_qaptiva_variational_qpu(device):
    try:
        from qlmaas.qpus import MPS
    except ImportError as error:
        raise ImportError(
            f"{device} requires QLMaaS configuration and myQLM's QLMaaS QPUs."
        ) from error

    from qat.vsolve.optimize import ScipyMinimizePlugin

    return ScipyMinimizePlugin(
        method="COBYLA",
        tol=1e-4,
        options={"maxiter": 300},
    ) | MPS()


def _build_qaptiva_mpstraj_qpu(params):
    steps = params.get("mpstraj_steps", 5000)
    bond_dimension = params.get("mpstraj_bond_dimension", 8)
    for name, value in (("mpstraj_steps", steps), ("mpstraj_bond_dimension", bond_dimension)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    try:
        from qlmaas.qpus import MPSTraj
    except ImportError as error:
        raise ImportError("MPSTraj requires a configured QLMaaS MPSTraj QPU.") from error
    return MPSTraj(n_steps=steps, bond_dimension=bond_dimension, seed=params["seed"])


def _build_qaptiva_annealing_qpu(device):
    try:
        from qlmaas.qpus import AnalogQPU, SQAQPU
    except ImportError as error:
        raise ImportError(
            f"{device} requires QLMaaS configuration and myQLM's QLMaaS QPUs."
        ) from error

    return {"AnalogQPU": AnalogQPU, "SQAQPU": SQAQPU}[device]()


def _build_qutip_qpu(params):
    steps = params.get("qutip_steps", 100)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("qutip_steps must be a positive integer.")
    try:
        from qat.analog.qutip_qpu import QutipQPU
    except ImportError as error:
        raise ImportError(
            "QutipQPU requires myQLM's analog/Qutip simulator dependencies."
        ) from error
    return QutipQPU(n_steps=steps)


def _build_simulated_annealing_qpu(params, seed):
    from qat.core.variables import Variable
    from qat.simulated_annealing import SimulatedAnnealing

    steps = params.get("annealing_steps", 50)
    temperature_min = params.get("temperature_min", 0.1)
    temperature_max = params.get("temperature_max", 5.0)
    shots = params.get("shots", annealing._DEFAULT_SHOTS)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("annealing_steps must be a positive integer.")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    if (
        isinstance(temperature_min, bool)
        or not isinstance(temperature_min, (int, float))
        or not math.isfinite(temperature_min)
        or temperature_min <= 0
    ):
        raise ValueError("temperature_min must be a finite positive number.")
    if (
        isinstance(temperature_max, bool)
        or not isinstance(temperature_max, (int, float))
        or not math.isfinite(temperature_max)
        or temperature_max <= temperature_min
    ):
        raise ValueError(
            "temperature_max must be finite and greater than temperature_min."
        )
    time = Variable("t", float)
    temperature = temperature_min * time + temperature_max * (1 - time)
    return SimulatedAnnealing(temp_t=temperature, n_steps=steps, seed=seed)


def _new_qscore(qpu, **kwargs):
    from qat.qscore.benchmark import QScore

    return QScore(qpu, **kwargs)


def _run_annealing(device, params):
    evolution_time = annealing._ANNEALING_TIME
    if device == "QutipQPU":
        qpu = _build_qutip_qpu(params)
        algorithm = "aqo"
        evolution_time = params.get("qutip_tmax", 5.0)
    elif device == "MPSTraj":
        qpu = _build_qaptiva_mpstraj_qpu(params)
        algorithm = "mpstraj"
        evolution_time = params.get("mpstraj_tmax", 1000.0)
    elif device in {"AnalogQPU", "SQAQPU"}:
        qpu = _build_qaptiva_annealing_qpu(device)
        algorithm = "aqo" if device == "AnalogQPU" else "sqa"
    else:
        qpu = _build_simulated_annealing_qpu(params, params["seed"])
        algorithm = "sa"
    return annealing.run(
        qpu,
        algorithm,
        initial_size=params["initial_size"],
        size_limit=params["size_limit"],
        beta=params["beta"],
        iterator=params["iterator"],
        seed=params["seed"],
        shots=params.get("shots", annealing._DEFAULT_SHOTS),
        aqo_time=evolution_time,
    )


def run(device, **params):
    """Run QScore with its QAOA core or one of the supported annealers."""
    if device not in _SUPPORTED_DEVICES:
        raise NotImplementedError(
            f"Unsupported QScore device {device!r}. "
            f"Choose one of {sorted(_SUPPORTED_DEVICES)}."
        )

    initial_size = params["initial_size"]
    size_limit = params["size_limit"]
    beta = params["beta"]
    iterator = params["iterator"]
    depth = params["depth"]
    seed = params["seed"]
    check_parameters_value(initial_size, size_limit, beta, iterator, depth, seed)

    if device in _QAOA_DEVICES:
        qpu = (
            _build_clinalg_qpu()
            if device == "CLinalg"
            else _build_qaptiva_variational_qpu(device)
        )
        with TemporaryDirectory(prefix="bacq-qscore-") as output_dir:
            output_path = Path(output_dir)
            report = _new_qscore(
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

        required_report_fields = {
            "status", "value", "beta", "evaluations", "raw_data"
        }
        if not isinstance(report, dict) or not required_report_fields.issubset(report):
            raise RuntimeError(
                "The installed qat-qscore version does not expose structured "
                "run results. Install the updated myqlm-compatibility-updates branch."
            )
    else:
        report = _run_annealing(device, params)

    run_parameters = {
        "initial_size": initial_size,
        "size_limit": size_limit,
        "beta": float(beta),
        "iterator": iterator,
        "depth": depth,
        "seed": seed,
    }
    if device == "QutipQPU":
        run_parameters["qutip_steps"] = params.get("qutip_steps", 100)
        run_parameters["qutip_tmax"] = params.get("qutip_tmax", 5.0)
    elif device == "MPSTraj":
        run_parameters.update(
            mpstraj_steps=params.get("mpstraj_steps", 5000),
            mpstraj_bond_dimension=params.get("mpstraj_bond_dimension", 8),
            mpstraj_tmax=params.get("mpstraj_tmax", 1000.0),
            shots=params.get("shots", annealing._DEFAULT_SHOTS),
        )
    elif device == "SimulatedAnnealing":
        run_parameters.update(
            annealing_steps=params.get("annealing_steps", 50),
            shots=params.get("shots", annealing._DEFAULT_SHOTS),
            temperature_min=params.get("temperature_min", 0.1),
            temperature_max=params.get("temperature_max", 5.0),
        )
    elif device == "SQAQPU":
        run_parameters["shots"] = params.get("shots", annealing._DEFAULT_SHOTS)

    return {
        "filetype": "experiment",
        "benchmark": "qscore",
        "device": device,
        "parameters": run_parameters,
        "result": report,
    }
