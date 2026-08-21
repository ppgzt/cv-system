"""Contrato congelado de Resource-Aware e das cinco modalidades PIBIC."""

import importlib.util
import pathlib
import unittest

from domain.pipeline_events import SelectionEvidenceEvent
from domain.resource_events import ResourceState, ResourceStateEvent
from domain.visual_activity import VisualState
from domain.visual_events import VisualStateEvent
from mas.agents.orchestrator_agent import OrchestratorAgent
from mas.agents.resource_manager_agent import ResourceManagerAgent, ResourceThresholds
from mas.experiment_config import EXPERIMENT_MODES
from pade.acl.aid import AID


def visual(index, state):
    return VisualStateEvent("P", index, float(index), float(index), None, state, None, 0.0)


def resource(sequence, state, observed_at=0):
    return ResourceStateEvent(sequence, observed_at, state, {})


class TestResourceAwareControl(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.orch = OrchestratorAgent(
            AID("o@localhost:1"), "c@localhost:2",
            resource_control_enabled=True,
            monotonic_ns=lambda: self.now,
        )
        self.orch.handle_passage_started("P")
        self.orch.handle_resource_state(resource(1, ResourceState.SAFE, self.now))

    def test_active_is_capped_by_resource_state(self):
        self.assertTrue(self.orch.handle_visual_state(visual(1, VisualState.ACTIVE)))
        self.assertEqual(self.orch.current_rate, "HIGH")
        self.assertTrue(self.orch.handle_resource_state(resource(2, ResourceState.WARNING, self.now)))
        self.assertEqual(self.orch.current_rate, "MEDIUM")
        self.assertTrue(self.orch.handle_resource_state(resource(3, ResourceState.CRITICAL, self.now)))
        self.assertEqual(self.orch.current_rate, "LOW")

    def test_recovery_recomputes_visual_interest(self):
        self.orch.handle_resource_state(resource(2, ResourceState.CRITICAL, self.now))
        self.orch.handle_visual_state(visual(1, VisualState.ACTIVE))
        self.orch.handle_resource_state(resource(3, ResourceState.WARNING, self.now))
        self.assertEqual(self.orch.current_rate, "MEDIUM")
        self.orch.handle_resource_state(resource(4, ResourceState.SAFE, self.now))
        self.assertEqual(self.orch.current_rate, "HIGH")
        self.orch.handle_visual_state(visual(2, VisualState.IDLE))
        self.assertEqual(self.orch.current_rate, "LOW")

    def test_hold_never_exceeds_cap_and_control_is_deduplicated(self):
        self.orch.handle_visual_state(visual(1, VisualState.ACTIVE))
        self.orch.handle_selection_evidence(SelectionEvidenceEvent("P", 1, "F", 10, True, .9))
        self.orch.handle_visual_state(visual(2, VisualState.IDLE))
        self.assertEqual(self.orch.current_rate, "HIGH")
        self.orch.handle_resource_state(resource(2, ResourceState.WARNING, self.now))
        self.assertEqual(self.orch.current_rate, "MEDIUM")
        self.orch.handle_resource_state(resource(3, ResourceState.CRITICAL, self.now))
        self.assertEqual(self.orch.current_rate, "LOW")
        self.assertFalse(self.orch.handle_resource_state(resource(3, ResourceState.SAFE, self.now)))
        self.assertFalse(self.orch.handle_visual_state(visual(2, VisualState.ACTIVE)))
        self.assertFalse(self.orch.handle_selection_evidence(SelectionEvidenceEvent("P", 2, "G", 10, False, .1)))

    def test_stale_safe_sample_never_authorizes_new_upshift_and_fresh_recovers(self):
        self.orch.handle_visual_state(visual(1, VisualState.IDLE))
        self.now += 10_000_000_001
        self.orch.handle_visual_state(visual(2, VisualState.ACTIVE))
        self.assertEqual(self.orch.current_rate, "LOW")
        self.assertFalse(self.orch._emit_control_state is None)
        self.assertTrue(self.orch.handle_resource_state(resource(2, ResourceState.SAFE, self.now)))
        self.assertEqual(self.orch.current_rate, "HIGH")

    def test_first_passage_has_no_artificial_low_cap_before_resource_sample(self):
        orch = OrchestratorAgent(
            AID("o@localhost:1"), "c@localhost:2",
            resource_control_enabled=True,
            monotonic_ns=lambda: self.now,
        )
        orch.handle_passage_started("P")
        self.assertTrue(orch.handle_visual_state(visual(1, VisualState.ACTIVE)))
        self.assertEqual(orch.current_rate, "HIGH")
        self.assertTrue(orch._resource_stale)

    def test_resource_events_are_ignored_outside_resource_aware_mode(self):
        orch = OrchestratorAgent(AID("o@localhost:1"), "c@localhost:2")
        orch.handle_passage_started("P")
        orch.handle_visual_state(visual(1, VisualState.ACTIVE))
        self.assertEqual(orch.current_rate, "HIGH")
        self.assertFalse(orch.handle_resource_state(resource(1, ResourceState.CRITICAL, 1)))
        self.assertEqual(orch.current_rate, "HIGH")


class TestResourceManagerClassification(unittest.TestCase):
    def setUp(self):
        self.manager = ResourceManagerAgent(AID("r@localhost:3"), "resource-test")

    def classify(self, *, temp, backlog=0, throttling_active=False):
        return self.manager._classify({
            "temperature_c": temp,
            "prediction_backlog": backlog,
            "throttling_active": throttling_active,
            "cpu_percent": 100.0,
            "ram_percent": 100.0,
        })

    def test_safe_at_74_9_without_backlog_or_throttle(self):
        self.assertEqual(self.classify(temp=74.9), (ResourceState.SAFE, []))

    def test_warning_temperature_at_75(self):
        state, reasons = self.classify(temp=75.0)
        self.assertIs(state, ResourceState.WARNING)
        self.assertEqual(reasons, ["temperature_c>=75.0"])

    def test_warning_backlog_at_7(self):
        state, reasons = self.classify(temp=20.0, backlog=7)
        self.assertIs(state, ResourceState.WARNING)
        self.assertEqual(reasons, ["prediction_backlog>=7"])

    def test_critical_temperature_at_80(self):
        state, reasons = self.classify(temp=80.0)
        self.assertIs(state, ResourceState.CRITICAL)
        self.assertEqual(reasons, ["temperature_c>=80.0"])

    def test_critical_throttling_and_precedence(self):
        state, reasons = self.classify(temp=75.0, backlog=7, throttling_active=True)
        self.assertIs(state, ResourceState.CRITICAL)
        self.assertEqual(reasons, ["throttling_active"])

    def test_temperature_none_is_not_zero_or_a_threshold_hit(self):
        state, reasons = self.classify(temp=None)
        self.assertIs(state, ResourceState.SAFE)
        self.assertEqual(reasons, [])

    def test_throttling_unavailable_stays_none(self):
        self.assertIsNone(self.manager.throttling_active(None))
        self.assertIsNone(self.manager.throttling_active({"throttling_command_available": False}))
        self.assertTrue(self.manager.throttling_active({
            "throttling_command_available": True,
            "throttled_current": True,
        }))
        self.assertFalse(self.manager.throttling_active({
            "throttling_command_available": True,
            "throttled_current": False,
            "undervoltage_current": False,
            "arm_frequency_capped_current": False,
            "soft_temperature_limit_current": False,
        }))

    def test_publish_sequences_and_send_failure_are_nonfatal(self):
        manager = ResourceManagerAgent(
            AID("r@localhost:3"), "resource-test",
            orchestrator_agent_aid="o@localhost:1", control_enabled=True,
        )
        manager.send = lambda _message: (_ for _ in ()).throw(RuntimeError("synthetic send failure"))
        snapshot = {"sampled_at_monotonic_ns": 5, "metrics": {
            "cpu_percent": 80.0, "ram_percent": 30.0, "temperature": None,
            "temperature_c": None, "prediction_backlog": 7, "throttling_active": None,
        }}
        first = manager.publish_resource_snapshot(snapshot)
        second = manager.publish_resource_snapshot(snapshot)
        self.assertEqual((first.sequence, second.sequence), (1, 2))
        self.assertIs(first.state, ResourceState.WARNING)
        self.assertEqual(first.metrics["temperature_c"], None)


class TestOfficialExperimentModes(unittest.TestCase):
    @staticmethod
    def _entrypoint_module():
        path = pathlib.Path(__file__).parents[1] / "mas-main.py"
        spec = importlib.util.spec_from_file_location("mas_main_mode_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_feature_matrix_is_exactly_the_five_official_conditions(self):
        self.assertEqual(set(EXPERIMENT_MODES), {
            "original-timing", "fixed-fps", "visual-adaptive", "visual-gated", "resource-aware-visual-gated",
        })
        self.assertFalse(EXPERIMENT_MODES["original-timing"].visual_adaptive)
        self.assertTrue(EXPERIMENT_MODES["fixed-fps"].requires_fixed_fps)
        self.assertTrue(EXPERIMENT_MODES["visual-adaptive"].visual_adaptive)
        self.assertFalse(EXPERIMENT_MODES["visual-adaptive"].visual_gated)
        self.assertTrue(EXPERIMENT_MODES["visual-gated"].visual_gated)
        self.assertFalse(EXPERIMENT_MODES["visual-gated"].resource_cap)
        self.assertTrue(EXPERIMENT_MODES["resource-aware-visual-gated"].resource_cap)

    def test_parser_selects_flags_and_frozen_defaults_for_each_mode(self):
        module = self._entrypoint_module()
        expected = {
            "original-timing": (None, True, False, False, False),
            "fixed-fps": (5.0, False, False, False, False),
            "visual-adaptive": (None, False, True, False, False),
            "visual-gated": (None, False, True, True, False),
            "resource-aware-visual-gated": (None, False, True, True, True),
        }
        for name, values in expected.items():
            argv = ["--engine", "pade", "--mode", name]
            if name == "fixed-fps":
                argv.extend(["--fps", "5"])
            args = module.build_parser().parse_args(argv)
            config = module.resolve_experiment_arguments(args, module.build_parser())
            self.assertEqual(
                (config["fps"], config["native_timestamps"], config["visual_event_enabled"], config["visual_gated"], config["resource_control_enabled"]),
                values,
            )
            self.assertEqual(args.low_fps, 4.0)
            self.assertEqual(args.medium_fps, 7.0)
            self.assertEqual(args.selector_threshold, 0.5)
            self.assertEqual(tuple(args.visual_roi), (0.30, 0.675, 0.00, 1.00))

    def test_fixed_fps_requires_explicit_fps(self):
        module = self._entrypoint_module()
        args = module.build_parser().parse_args(["--engine", "pade", "--mode", "fixed-fps"])
        with self.assertRaises(SystemExit):
            module.resolve_experiment_arguments(args, module.build_parser())


if __name__ == "__main__":
    unittest.main()
