import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "data-analysis"
    / "visual_event_relabeling"
    / "build_relabeling_queue.py"
)
SPEC = importlib.util.spec_from_file_location("visual_event_relabeling", MODULE_PATH)
workflow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(workflow)


def review_row(passage_id="passage-b", capture_index=1, final_review=""):
    return {
        "review_order": "1",
        "passage_id": passage_id,
        "capture_index": str(capture_index),
        "relative_time_ms": "0",
        "rgb_filename": "rgb.png",
        "rgb_prev_3": "",
        "rgb_prev_2": "",
        "rgb_prev_1": "",
        "rgb_next_1": "",
        "rgb_next_2": "",
        "rgb_next_3": "",
        "original_label": "background",
        "candidate_reason": "boundary",
        "final_review": final_review,
        "notes": "",
    }


class VisualEventRelabelingScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passage_ids, cls.indexes, _ = workflow.load_operational_indexes(
            workflow.base.DEFAULT_DATA_ROOT,
            workflow.base.DEFAULT_COHORT_METRICS,
        )
        cls.candidates = workflow.build_candidates(
            cls.indexes,
            workflow.label_audit.pair_lookup(workflow.DEFAULT_PAIR_FEATURES),
        )
        cls.manifest = workflow.build_manifest(cls.candidates, cls.indexes)

    def test_exact_scope_is_background_only_deduplicated_and_operational(self):
        workflow.assert_expected_scope(self.candidates)
        keys = [
            (row["passage_id"], int(row["capture_index"]))
            for row in self.candidates
        ]
        self.assertEqual(1_301, len(keys))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(row["original_label"] == "background" for row in self.candidates))
        self.assertEqual(set(self.passage_ids), {row["passage_id"] for row in self.candidates})
        self.assertEqual(
            {
                "boundary": 1_077,
                "high_pdi": 146,
                "boundary+high_pdi": 78,
            },
            dict(Counter(row["candidate_reason"] for row in self.candidates)),
        )

    def test_main_queue_is_ordered_and_rgb_references_match_index_metadata(self):
        keys = [
            (row["passage_id"], int(row["capture_index"]))
            for row in self.manifest
        ]
        self.assertEqual(sorted(keys), keys)
        for row in self.manifest:
            source = self.indexes[row["passage_id"]][int(row["capture_index"]) - 1]
            self.assertEqual(source["rgb_filename"], row["rgb_filename"])
            self.assertNotIn("depth_review", row)
            self.assertNotIn("depth_panel", row)

    def test_pilot_is_150_rgb_candidates_stratified_and_passage_ordered(self):
        pilot = workflow.build_pilot(self.manifest)
        self.assertEqual(150, len(pilot))
        self.assertEqual(
            {"boundary": 50, "high_pdi": 50, "boundary+high_pdi": 50},
            dict(Counter(row["candidate_reason"] for row in pilot)),
        )
        keys = [(row["passage_id"], int(row["capture_index"])) for row in pilot]
        self.assertEqual(sorted(keys), keys)
        self.assertGreater(len({row["passage_id"] for row in pilot}), 100)

    def test_passage_summary_covers_all_operational_passages(self):
        summary = workflow.build_passage_summary(self.manifest)
        self.assertEqual(184, len(summary))
        self.assertEqual(1_301, sum(row["candidate_count"] for row in summary))
        for row in summary:
            self.assertLessEqual(row["min_capture_index"], row["max_capture_index"])


class VisualEventRelabelingWorkflowTests(unittest.TestCase):
    def test_neighbor_references_never_cross_passage_boundaries(self):
        indexes = {
            "passage-a": [
                {"rgb_filename": "a0.jpg"},
                {"rgb_filename": "a1.jpg"},
            ],
            "passage-b": [
                {"rgb_filename": "b0.jpg"},
                {"rgb_filename": "b1.jpg"},
            ],
        }
        candidate = {
            "passage_id": "passage-b",
            "capture_index": 1,
            "relative_time_ms": 0.0,
            "rgb_filename": "b0.jpg",
            "original_label": "background",
            "candidate_reason": "boundary",
        }
        row = workflow.build_manifest([candidate], indexes)[0]
        self.assertEqual("", row["rgb_prev_1"])
        self.assertEqual("b1.jpg", row["rgb_next_1"])
        self.assertNotIn("a1.jpg", row.values())

    def test_building_queue_requires_only_rgb_metadata_not_rgb_or_depth_files(self):
        indexes = {
            "passage-a": [
                {
                    "relative_time_ms": 0,
                    "rgb_filename": "not-local-0.jpg",
                    "label": "background",
                },
                {
                    "relative_time_ms": 100,
                    "rgb_filename": "not-local-1.jpg",
                    "label": "parcial",
                },
            ]
        }
        candidates = workflow.build_candidates(indexes, pairs={})
        manifest = workflow.build_manifest(candidates, indexes)
        self.assertEqual(1, len(manifest))
        self.assertEqual("not-local-0.jpg", manifest[0]["rgb_filename"])
        self.assertNotIn("depth_filename", manifest[0])

    def test_only_three_final_human_labels_are_accepted(self):
        self.assertEqual(
            {"", "CLEAR_EMPTY", "ANIMAL_VISIBLE", "AMBIGUOUS"},
            set(workflow.FINAL_REVIEW_VALUES),
        )
        self.assertNotIn("NEEDS_RGB", workflow.FINAL_REVIEW_VALUES)

    def test_consolidation_maps_final_labels_and_skips_unreviewed(self):
        manifest = [
            review_row("p", 1, ""),
            review_row("p", 2, "CLEAR_EMPTY"),
            review_row("p", 3, "ANIMAL_VISIBLE"),
            review_row("p", 4, "AMBIGUOUS"),
        ]
        output = workflow.consolidate_relabels(manifest)
        self.assertEqual([2, 3, 4], [int(row["capture_index"]) for row in output])
        self.assertEqual(
            ["NEGATIVE", "POSITIVE", "EXCLUDE"],
            [row["visual_event_label"] for row in output],
        )
        self.assertEqual(["YES", "YES", "NO"], [row["training_eligibility"] for row in output])

    def test_needs_rgb_cannot_enter_consolidated_ground_truth(self):
        with self.assertRaisesRegex(ValueError, "cannot consolidate final label"):
            workflow.consolidate_relabels([review_row("passage-a", 1, "NEEDS_RGB")])

    def test_candidate_building_does_not_modify_original_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulation_index.json"
            rows = [
                {
                    "relative_time_ms": 0,
                    "rgb_filename": "r0.jpg",
                    "label": "background",
                },
                {
                    "relative_time_ms": 100,
                    "rgb_filename": "r1.jpg",
                    "label": "parcial",
                },
            ]
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            before = path.read_bytes()
            candidates = workflow.build_candidates({"p": rows}, pairs={})
            workflow.build_manifest(candidates, {"p": rows})
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
