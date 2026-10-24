"""Run QScore's MaxCut size search with myQLM annealing QPUs."""

from numbers import Integral, Real

import networkx as nx
import numpy as np
from qat.opt import MaxCut
from qat.qscore.iterators import Driver


_NB_INSTANCES_PER_SIZE = 100
_ANNEALING_TIME = 1.0
_DEFAULT_SHOTS = 100


def _state_bits(state, n_qubits):
    if isinstance(state, bool):
        raise TypeError("QPU sample state must not be a boolean.")
    if isinstance(state, Integral):
        integer_state = int(state)
        return [(integer_state >> index) & 1 for index in range(n_qubits)]

    state_text = str(state).replace("|", "").replace(">", "").strip()
    if state_text.startswith("0b"):
        state_text = state_text[2:]
    if len(state_text) == n_qubits and set(state_text).issubset({"0", "1"}):
        return [int(bit) for bit in state_text[::-1]]

    raise TypeError(f"Unsupported QPU sample state: {state!r}.")


def _cut_score(graph, state):
    bits = _state_bits(state, graph.number_of_nodes())
    return float(sum(bits[left] != bits[right] for left, right in graph.edges()))


def _score_from_result(result, graph):
    value = getattr(result, "value", None)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("QPU result value must be a real number or null.")
        return float(-value)

    raw_data = getattr(result, "raw_data", None)
    if raw_data is None:
        raise RuntimeError("Annealing QPU result contains neither a value nor samples.")
    samples = list(raw_data)
    if not samples:
        raise RuntimeError("Annealing QPU returned no samples.")
    return max(_cut_score(graph, sample.state) for sample in samples)


def _make_job(problem, algorithm, shots):
    if algorithm == "aqo":
        return problem.aqo_job(tmax=_ANNEALING_TIME, nbshots=0)
    if algorithm == "sa":
        return problem.sqa_job(tmax=_ANNEALING_TIME, nbshots=shots)
    raise ValueError(f"Unsupported annealing algorithm: {algorithm!r}.")


def run(
    qpu,
    algorithm,
    *,
    initial_size,
    size_limit,
    beta,
    iterator,
    seed,
    shots=_DEFAULT_SHOTS,
):
    """Execute the same graph-size QScore search using AQO or SA jobs."""
    all_data = {}
    evaluations = {}
    next_seed = seed if seed is not None else int(np.random.randint(100000))

    def evaluate_size(size):
        nonlocal next_seed
        instances = []
        scores = []

        for _ in range(_NB_INSTANCES_PER_SIZE):
            current_seed = next_seed
            next_seed += 1
            graph = nx.generators.erdos_renyi_graph(size, 0.5, seed=current_seed)
            graph_seed = current_seed
            if algorithm == "aqo" and graph.number_of_edges() == 0:
                retry_seed = current_seed
                for _ in range(10):
                    retry_seed += 1
                    graph = nx.generators.erdos_renyi_graph(
                        size, 0.5, seed=retry_seed
                    )
                    if graph.number_of_edges():
                        graph_seed = retry_seed
                        break
                else:
                    raise RuntimeError(
                        f"Could not generate a non-empty graph of size {size}."
                    )

            next_seed = max(next_seed, graph_seed + 1)
            problem = MaxCut(graph)
            job = _make_job(problem, algorithm, shots)
            result = qpu.submit(job)
            score = _score_from_result(result, graph)
            scores.append(score)
            instances.append({"seed": graph_seed, "score": score})

        average_score = float(np.mean(scores) - size * (size - 1) / 8)
        random_best_score = float(0.178 * size ** 1.5)
        ratio = average_score / random_best_score
        all_data[size] = instances
        evaluations[size] = {
            "average_score": average_score,
            "random_best_score": random_best_score,
            "ratio": ratio,
            "success": ratio > beta,
        }
        print(
            f"{algorithm.upper()} n={size}: score={average_score:.2f}, "
            f"random best={random_best_score:.2f}, ratio={ratio:.3f}"
        )
        return ratio - beta

    success, _, info = Driver(
        evaluate_size, iterator, initial_size, size_limit
    ).run()
    if success:
        status, value = "boundary", info
    elif info[0]:
        status, value = "above_limit", info[1]
    else:
        status, value = "below_initial", info[1]

    return {
        "status": status,
        "value": int(value),
        "beta": float(beta),
        "evaluations": evaluations,
        "raw_data": all_data,
    }
