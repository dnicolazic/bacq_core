"""QScore integration tests; requires the project's [qscore] extra."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bacq_core.benchmarks.qscore import evaluate, experiment, helpers_protocol


VALID_PARAMETERS = {
    "initial_size": 2,
    "size_limit": 4,
    "beta": 0.2,
    "iterator": "dichotomic",
    "depth": 1,
    "seed": 42,
}

VALID_REPORT = {
    "status": "boundary",
    "value": 3,
    "beta": 0.2,
    "evaluations": {},
    "raw_data": {},
}


class ProtocolTests(unittest.TestCase):
    def test_valid_parameters(self):
        helpers_protocol.check_parameters_value(**VALID_PARAMETERS)

    def test_null_seed_is_valid(self):
        params = {**VALID_PARAMETERS, "seed": None}
        helpers_protocol.check_parameters_value(**params)

    def test_invalid_numeric_and_search_parameters_are_rejected(self):
        invalid_cases = (
            ("initial_size", 1, ValueError),
            ("initial_size", True, ValueError),
            ("size_limit", 1, ValueError),
            ("size_limit", True, TypeError),
            ("beta", 0, ValueError),
            ("beta", 1, ValueError),
            ("beta", float("nan"), TypeError),
            ("beta", float("inf"), TypeError),
            ("beta", True, TypeError),
            ("iterator", "unknown", ValueError),
            ("iterator", None, ValueError),
            ("depth", 0, ValueError),
            ("depth", True, ValueError),
            ("seed", True, TypeError),
            ("seed", "42", TypeError),
        )
        for name, value, error in invalid_cases:
            with self.subTest(parameter=name, value=value):
                params = {**VALID_PARAMETERS, name: value}
                with self.assertRaises(error):
                    helpers_protocol.check_parameters_value(**params)

    def test_size_limit_must_not_be_less_than_initial_size(self):
        params = {**VALID_PARAMETERS, "initial_size": 4, "size_limit": 3}
        with self.assertRaisesRegex(ValueError, "size_limit"):
            helpers_protocol.check_parameters_value(**params)

    def test_qscore_does_not_accept_bacq_sequence_protocols(self):
        with self.assertRaisesRegex(NotImplementedError, "protocol_seq"):
            helpers_protocol.sequence_generator()


class EvaluationTests(unittest.TestCase):
    def test_boundary_and_bounds_map_to_bacq_metric_and_success(self):
        cases = (
            ("boundary", True),
            ("above_limit", False),
            ("below_initial", False),
        )
        for status, success in cases:
            with self.subTest(status=status):
                data = {"result": {**VALID_REPORT, "status": status}}
                self.assertEqual(evaluate.get_metric_success(data), (3, success))

    def test_unknown_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown QScore result status"):
            evaluate.get_metric_success(
                {"result": {**VALID_REPORT, "status": "unknown"}}
            )

    def test_non_integer_metric_is_rejected(self):
        for value in (True, 3.0, "3"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                evaluate.get_metric_success(
                    {"result": {**VALID_REPORT, "value": value}}
                )

    def test_sequence_scoring_is_not_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "graph-size search"):
            evaluate.get_score({})


class ExperimentTests(unittest.TestCase):
    def _run_with_mocks(self, report=VALID_REPORT):
        emulator = MagicMock(name="CLinalg instance")
        optimizer = MagicMock(name="optimizer")
        composed_qpu = MagicMock(name="composed QPU")
        optimizer.__or__.return_value = composed_qpu
        qscore = MagicMock(name="QScore instance")
        qscore.run.return_value = report

        with (
            patch.object(experiment, "CLinalg", return_value=emulator) as clinalg,
            patch.object(
                experiment, "ScipyMinimizePlugin", return_value=optimizer
            ) as plugin,
            patch.object(experiment, "QScore", return_value=qscore) as qscore_class,
        ):
            result = experiment.run("CLinalg", **VALID_PARAMETERS)

        return result, emulator, optimizer, composed_qpu, qscore, clinalg, plugin, qscore_class

    def test_runs_qscore_with_clinalg_and_returns_bacq_experiment(self):
        (
            result,
            emulator,
            optimizer,
            composed_qpu,
            qscore,
            clinalg,
            plugin,
            qscore_class,
        ) = self._run_with_mocks()

        self.assertEqual(result["filetype"], "experiment")
        self.assertEqual(result["benchmark"], "qscore")
        self.assertEqual(result["device"], "CLinalg")
        self.assertEqual(result["result"], VALID_REPORT)
        self.assertEqual(result["parameters"], VALID_PARAMETERS)
        clinalg.assert_called_once_with()
        plugin.assert_called_once_with(
            method="COBYLA", tol=1e-4, options={"maxiter": 300}
        )
        optimizer.__or__.assert_called_once_with(emulator)
        qscore_class.assert_called_once()
        self.assertIs(qscore_class.call_args.args[0], composed_qpu)
        qscore.run.assert_called_once_with()

    def test_qscore_output_files_use_temporary_directory(self):
        _, _, _, _, _, _, _, qscore_class = self._run_with_mocks()
        kwargs = qscore_class.call_args.kwargs
        self.assertTrue(kwargs["output"].endswith("/qscore.csv"))
        self.assertTrue(kwargs["rawdata"].endswith("/qscore.raw"))
        self.assertIn("bacq-qscore-", kwargs["output"])
        self.assertEqual(kwargs["initial_size"], VALID_PARAMETERS["initial_size"])
        self.assertEqual(kwargs["size_limit"], VALID_PARAMETERS["size_limit"])
        self.assertEqual(kwargs["seed"], VALID_PARAMETERS["seed"])
        self.assertFalse(Path(kwargs["output"]).parent.exists())

    def test_other_devices_are_not_supported(self):
        with patch.object(experiment, "QScore") as qscore_class:
            with self.assertRaisesRegex(NotImplementedError, "CLinalg"):
                experiment.run("PyLinalg", **VALID_PARAMETERS)
        qscore_class.assert_not_called()

    def test_invalid_parameters_do_not_start_qscore(self):
        params = {**VALID_PARAMETERS, "depth": 0}
        with patch.object(experiment, "QScore") as qscore_class:
            with self.assertRaises(ValueError):
                experiment.run("CLinalg", **params)
        qscore_class.assert_not_called()

    def test_incompatible_qscore_result_api_has_clear_error(self):
        old_api_report = {"status": "boundary"}
        with patch.object(experiment, "QScore") as qscore_class:
            qscore_class.return_value.run.return_value = old_api_report
            with self.assertRaisesRegex(RuntimeError, "structured run results"):
                experiment.run("CLinalg", **VALID_PARAMETERS)

    def test_small_qscore_run_executes_on_real_clinalg_emulator(self):
        with patch("qat.qscore.benchmark._NB_INSTANCES_PER_SIZE", 1):
            result = experiment.run(
                "CLinalg",
                initial_size=2,
                size_limit=2,
                beta=0.2,
                iterator="exhaustive",
                depth=1,
                seed=42,
            )

        self.assertEqual(result["filetype"], "experiment")
        self.assertEqual(result["benchmark"], "qscore")
        self.assertEqual(result["device"], "CLinalg")
        self.assertIn(result["result"]["status"], {"above_limit", "below_initial"})
        self.assertEqual(len(result["result"]["raw_data"][2]), 1)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
