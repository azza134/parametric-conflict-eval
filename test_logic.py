# Tests to verify the functions are working as intended. No API key required.
import math
import unittest

from harness import grade, perturb, wilsons
from judge import (cohens_kappa, is_faithful, FAITHFUL, LEAK,
                   judge_gate, anchor_disagreements, GATE_PASS, GATE_FAIL, KAPPA_THRESHOLD)


class TestGrade(unittest.TestCase):
    def test_must_contain_present(self):
        self.assertTrue(grade("the rate is one per million", must_contain="million"))

    def test_must_contain_absent(self):
        self.assertFalse(grade("no figure here", must_contain="million"))

    def test_must_not_contain_blocks(self):
        self.assertFalse(grade("about 20 people", must_not_contain="20"))

    def test_word_boundary(self):
        # "20" must NOT match inside "2023". Whole-word matching only
        self.assertTrue(grade("approved in 2023", must_not_contain="20"))

    def test_case_insensitive(self):
        self.assertTrue(grade("NOT IN DOCUMENT", must_contain="not in document"))

    def test_no_rules_passes(self):
        self.assertTrue(grade("anything at all")) # When no requirements are given a pass should be recorded


class TestPerturb(unittest.TestCase):
    def test_replaces(self):
        self.assertEqual(perturb("every 20 persons", [("every 20", "every 13")]), "every 13 persons") # when 20 is replaced with 13 does it equal argument 3

    def test_raises_on_noop(self):
        with self.assertRaises(AssertionError):
            perturb("nothing to change here", [("absent token", "x")]) # argument 1 is a phrase that doesn't exist in passage, thus nothing to replace, raising the assertion error for the perturb function


class TestWilsons(unittest.TestCase):
    def test_point_estimate(self):
        p, low, high = wilsons(3, 5)
        self.assertAlmostEqual(p, 0.6)
        self.assertTrue(0.0 <= low <= p <= high <= 1.0)

    def test_extreme_is_clamped_and_honest(self):
        p, low, high = wilsons(5, 5)
        self.assertEqual(p, 1.0)
        self.assertEqual(high, 1.0)   
        self.assertLess(low, 1.0)     # prevents [1.00, 1.00] that Wald CI gives


class TestCohensKappa(unittest.TestCase):
    def test_perfect_with_variation(self):
        po, k = cohens_kappa([True, False, True, False], [True, False, True, False])
        self.assertEqual(po, 1.0)
        self.assertAlmostEqual(k, 1.0)

    def test_all_one_class_is_nan(self):
        po, k = cohens_kappa([True, True, True], [True, True, True])
        self.assertEqual(po, 1.0)
        self.assertTrue(math.isnan(k))

    def test_chance_level(self):
        po, k = cohens_kappa([True, True, False, False], [True, False, True, False])
        self.assertAlmostEqual(po, 0.5)
        self.assertAlmostEqual(k, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cohens_kappa([True], [True, False])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cohens_kappa([], [])


class TestLabels(unittest.TestCase):
    def test_is_faithful_maps_words_to_bools(self):
        self.assertTrue(is_faithful(FAITHFUL))
        self.assertFalse(is_faithful(LEAK))


class TestJudgeGate(unittest.TestCase):
    # the gate only reads role/human/judge off each row, so a tiny factory is enough
    def _row(self, role, human, judge):
        return {"role": role, "human": human, "judge": judge}

    def _agree_anchor(self):  # an anchor row the judge got right (keeps PRIMARY gate non-vacuous)
        return self._row("clean-leak anchor", FAITHFUL, FAITHFUL)

    def test_clean_anchors_high_kappa_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], 0.95)
        self.assertEqual(verdict, GATE_PASS)

    def test_anchor_disagreement_fails_despite_high_kappa(self):
        # PRIMARY dominates: an obvious-case miss fails even with perfect aggregate agreement
        verdict, reasons = judge_gate([self._row("clean-leak anchor", FAITHFUL, LEAK)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("anchor" in r.lower() for r in reasons))

    def test_low_kappa_with_clean_anchors_fails(self):
        verdict, reasons = judge_gate([self._agree_anchor()], 0.5)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("kappa" in r.lower() for r in reasons))

    def test_borderline_disagreement_only_passes(self):
        rows = [self._agree_anchor(), self._row("borderline", FAITHFUL, LEAK)]
        verdict, _ = judge_gate(rows, 0.9)
        self.assertEqual(verdict, GATE_PASS)  # borderline disagreements are allowed

    def test_nan_kappa_clean_anchors_passes_with_warning(self):
        verdict, reasons = judge_gate([self._agree_anchor()], float("nan"))
        self.assertEqual(verdict, GATE_PASS)
        self.assertTrue(any("WARNING" in r for r in reasons))

    def test_nan_kappa_anchor_disagreement_still_fails(self):
        # NaN must not mask an anchor miss
        verdict, _ = judge_gate([self._row("clean-faithful anchor", FAITHFUL, LEAK)], float("nan"))
        self.assertEqual(verdict, GATE_FAIL)

    def test_kappa_exactly_threshold_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], KAPPA_THRESHOLD)  # pins inclusive >=
        self.assertEqual(verdict, GATE_PASS)

    def test_anchor_matching_is_substring(self):
        rows = [self._row("clean-leak anchor", FAITHFUL, LEAK),
                self._row("clean-faithful anchor", FAITHFUL, LEAK),
                self._row("borderline", FAITHFUL, LEAK)]
        bad = anchor_disagreements(rows)
        self.assertEqual(len(bad), 2)  # both anchors, not the borderline
        self.assertTrue(all("anchor" in r["role"] for r in bad))

    def test_no_anchor_rows_fails(self):
        # fail-closed: only borderline rows -> PRIMARY gate is vacuous
        verdict, reasons = judge_gate([self._row("borderline", FAITHFUL, FAITHFUL)], 0.95)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("vacuous" in r.lower() or "anchor" in r.lower() for r in reasons))


if __name__ == "__main__":
    unittest.main()
