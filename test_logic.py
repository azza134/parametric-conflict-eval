# Tests to verify the functions are working as intended. No API key required.
import json
import math
import os
import tempfile
import unittest
from unittest import mock

from config import (perturb, with_retry, MODELS, FLAG_INVITING, SYSTEM_INSTRUCTIONS,
                    appears, passage, step_doc)
from harness import (wilson_interval, PERTURBATION_LADDERS, SEVERITIES, validate_ladders,
                     total_steps, classify, lexical_caveat, UNANSWERABLE_ITEMS, validate_items,
                     load_done, tradeoff_rows, PRIOR_STRENGTHS)
from judge import (cohens_kappa, FAITHFUL, UNGROUNDED,
                   judge_gate, anchor_disagreements, GATE_PASS, GATE_FAIL, KAPPA_THRESHOLD,
                   QUESTIONED, SILENT, ENDORSED, CAVEAT_LABELS, CAVEAT_SCHEMA, build_caveat_prompt,
                   gold_schedule, _meta_evaluate)


class TestPerturb(unittest.TestCase):
    def test_replaces(self):
        self.assertEqual(perturb("every 20 persons", [("every 20", "every 13")]), "every 13 persons") # when 20 is replaced with 13 does it equal argument 3

    def test_raises_on_noop(self):
        with self.assertRaises(AssertionError):
            perturb("nothing to change here", [("absent token", "x")]) # argument 1 is a phrase that doesn't exist in passage, thus nothing to replace, raising the assertion error for the perturb function


class TestWilsonInterval(unittest.TestCase):
    def test_point_estimate(self):
        p, low, high = wilson_interval(3, 5)
        self.assertAlmostEqual(p, 0.6)
        self.assertTrue(0.0 <= low <= p <= high <= 1.0)

    def test_extreme_is_clamped_and_honest(self):
        p, low, high = wilson_interval(5, 5)
        self.assertEqual(p, 1.0)
        self.assertEqual(high, 1.0)   
        self.assertLess(low, 1.0)     # prevents [1.00, 1.00] that Wald CI gives


class TestCohensKappa(unittest.TestCase):
    def test_perfect_three_class(self):
        po, k = cohens_kappa([QUESTIONED, SILENT, ENDORSED], [QUESTIONED, SILENT, ENDORSED])
        self.assertEqual(po, 1.0)
        self.assertAlmostEqual(k, 1.0)

    def test_one_class_is_nan(self):
        po, k = cohens_kappa([SILENT, SILENT], [SILENT, SILENT])
        self.assertEqual(po, 1.0)
        self.assertTrue(math.isnan(k))

    def test_two_class_chance_level(self):
        po, k = cohens_kappa([FAITHFUL, FAITHFUL, UNGROUNDED, UNGROUNDED],
                            [FAITHFUL, UNGROUNDED, FAITHFUL, UNGROUNDED])
        self.assertAlmostEqual(po, 0.5)
        self.assertAlmostEqual(k, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cohens_kappa([SILENT], [SILENT, ENDORSED])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cohens_kappa([], [])



class TestJudgeGate(unittest.TestCase):
    # the gate only reads role/human/judge off each row, so a tiny factory is enough
    def _row(self, role, human, judge):
        return {"role": role, "human": human, "judge": judge}

    def _agree_anchor(self):  # an anchor row the judge got right (keeps PRIMARY gate non-vacuous)
        return self._row("clean-ungrounded anchor", FAITHFUL, FAITHFUL)

    def test_clean_anchors_high_kappa_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], 0.95)
        self.assertEqual(verdict, GATE_PASS)

    def test_anchor_disagreement_fails_despite_high_kappa(self):
        # PRIMARY dominates: an obvious-case miss fails even with perfect aggregate agreement
        verdict, reasons = judge_gate([self._row("clean-ungrounded anchor", FAITHFUL, UNGROUNDED)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("anchor" in r.lower() for r in reasons))

    def test_low_kappa_with_clean_anchors_fails(self):
        verdict, reasons = judge_gate([self._agree_anchor()], 0.5)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("kappa" in r.lower() for r in reasons))

    def test_borderline_disagreement_only_passes(self):
        rows = [self._agree_anchor(), self._row("borderline", FAITHFUL, UNGROUNDED)]
        verdict, _ = judge_gate(rows, 0.9)
        self.assertEqual(verdict, GATE_PASS)  # borderline disagreements are allowed

    def test_nan_kappa_clean_anchors_passes_with_warning(self):
        verdict, reasons = judge_gate([self._agree_anchor()], float("nan"))
        self.assertEqual(verdict, GATE_PASS)
        self.assertTrue(any("WARNING" in r for r in reasons))

    def test_nan_kappa_anchor_disagreement_still_fails(self):
        # NaN must not mask an anchor miss
        verdict, _ = judge_gate([self._row("clean-faithful anchor", FAITHFUL, UNGROUNDED)], float("nan"))
        self.assertEqual(verdict, GATE_FAIL)

    def test_kappa_exactly_threshold_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], KAPPA_THRESHOLD)  # pins inclusive >=
        self.assertEqual(verdict, GATE_PASS)

    def test_anchor_matching_is_substring(self):
        rows = [self._row("clean-ungrounded anchor", FAITHFUL, UNGROUNDED),
                self._row("clean-faithful anchor", FAITHFUL, UNGROUNDED),
                self._row("borderline", FAITHFUL, UNGROUNDED)]
        bad = anchor_disagreements(rows)
        self.assertEqual(len(bad), 2)  # both anchors, not the borderline
        self.assertTrue(all("anchor" in r["role"] for r in bad))

    def test_no_anchor_rows_fails(self):
        # fail-closed: only borderline rows -> PRIMARY gate is vacuous
        verdict, reasons = judge_gate([self._row("borderline", FAITHFUL, FAITHFUL)], 0.95)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("vacuous" in r.lower() or "anchor" in r.lower() for r in reasons))


class TestCaveatGate(unittest.TestCase):
    def _row(self, role, human, judge):
        return {"role": role, "human": human, "judge": judge}

    def _agree_anchor(self):
        return self._row("extreme-perturbation anchor", QUESTIONED, QUESTIONED)

    def test_caveat_anchor_agreement_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], 0.9)
        self.assertEqual(verdict, GATE_PASS)

    def test_questioned_anchor_miss_fails_despite_high_kappa(self):
        verdict, _ = judge_gate([self._row("extreme-perturbation anchor", QUESTIONED, SILENT)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)

    def test_silent_anchor_miss_fails(self):
        verdict, _ = judge_gate([self._row("extreme-perturbation anchor", SILENT, ENDORSED)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)

    def test_subtle_borderline_disagreement_passes(self):
        rows = [self._agree_anchor(), self._row("subtle borderline", SILENT, ENDORSED)]
        verdict, _ = judge_gate(rows, 0.9)
        self.assertEqual(verdict, GATE_PASS)

    def test_no_anchor_rows_fails(self):
        verdict, _ = judge_gate([self._row("subtle borderline", QUESTIONED, QUESTIONED)], 0.95)
        self.assertEqual(verdict, GATE_FAIL)

    def test_anchor_disagreements_picks_extreme_anchor(self):
        rows = [self._row("extreme-perturbation anchor", QUESTIONED, SILENT),
                self._row("subtle borderline", QUESTIONED, SILENT)]
        bad = anchor_disagreements(rows)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["role"], "extreme-perturbation anchor")


class TestCaveatPrompt(unittest.TestCase):
    def test_prompt_contains_its_two_args(self):
        prompt = build_caveat_prompt("How tall is X?", "It is 500 metres.")
        self.assertIn("How tall is X?", prompt)
        self.assertIn("It is 500 metres.", prompt)

    def test_prompt_takes_no_document(self):
        with self.assertRaises(TypeError):
            build_caveat_prompt("q", "a", "the correct value is 10cm")


class TestCaveatSchema(unittest.TestCase):
    def test_schema_shape(self):
        self.assertEqual(CAVEAT_SCHEMA["required"], ["stance", "reason"])
        self.assertIs(CAVEAT_SCHEMA["additionalProperties"], False)
        self.assertEqual(CAVEAT_SCHEMA["properties"]["stance"]["enum"], list(CAVEAT_LABELS))

    def test_labels_are_the_three_stances(self):
        self.assertEqual(CAVEAT_LABELS, (QUESTIONED, SILENT, ENDORSED))


class TestWithRetry(unittest.TestCase):
    def test_returns_on_first_success(self):
        calls = []
        def fn(x):
            calls.append(x)
            return x * 2
        self.assertEqual(with_retry(fn, 5), 10)
        self.assertEqual(len(calls), 1)

    def test_raises_after_exhausting_attempts(self):
        def fn():
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            with_retry(fn, attempts=1)

    def test_retries_then_succeeds(self):
        state = {"n": 0}
        def fn():
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("transient")
            return "ok"
        with mock.patch("config.time.sleep"):
            self.assertEqual(with_retry(fn, attempts=5), "ok")
        self.assertEqual(state["n"], 3)


class TestInstructions(unittest.TestCase):
    def test_instruction_names_and_order(self):
        self.assertEqual([name for name, _ in SYSTEM_INSTRUCTIONS], ["SOURCE_EXCLUSIVE", "FLAG_INVITING", "WEAK_GROUNDING"])

    def test_three_distinct_instructions(self):
        self.assertEqual(len({t for _, t in SYSTEM_INSTRUCTIONS}), 3)

    def test_permissive_invites_flagging(self):
        self.assertIn("flag", FLAG_INVITING.lower())


class TestLadders(unittest.TestCase):
    def test_all_perturbations_apply_to_document(self):
        self.assertEqual(validate_ladders(), [])

    def test_every_fact_ladders_0_to_5(self):
        for fact in PERTURBATION_LADDERS:
            self.assertEqual([s["severity"] for s in fact["steps"]], SEVERITIES)

    def test_thirty_six_steps_total(self):
        self.assertEqual(total_steps(), 36)

    def test_ratio_increases_where_present(self):
        for fact in PERTURBATION_LADDERS:
            ratios = [s["ratio"] for s in fact["steps"] if s["ratio"] is not None]
            self.assertEqual(ratios, sorted(ratios))

    def test_saturday_hours_is_non_ratio(self):
        sat = next(f for f in PERTURBATION_LADDERS if f["fact"] == "saturday_hours")
        self.assertTrue(all(s["ratio"] is None for s in sat["steps"]))

    def test_noop_perturbation_reported_via_exception_alone(self):
        with mock.patch("harness.perturb", side_effect=AssertionError("no change in passage detected")):
            problems = validate_ladders()
        self.assertTrue(any("no change in passage detected" in p for p in problems))
        self.assertEqual(len(problems), total_steps() - len(PERTURBATION_LADDERS))


class TestControlRung(unittest.TestCase):
    def test_every_fact_leads_with_an_unperturbed_control(self):
        for fact in PERTURBATION_LADDERS:
            first = fact["steps"][0]
            self.assertEqual(first["severity"], 0, fact["fact"])
            self.assertEqual(first["replace"], [], fact["fact"])

    def test_control_target_strings_present_in_document(self):
        for fact in PERTURBATION_LADDERS:
            self.assertTrue(appears(fact["steps"][0]["target_string"], passage), fact["fact"])

    def test_step_doc_control_is_the_real_passage(self):
        self.assertEqual(step_doc({"replace": []}), passage)

    def test_step_doc_applies_perturbation(self):
        doc = step_doc({"replace": [("every 20", "every 13")]})
        self.assertNotEqual(doc, passage)
        self.assertIn("every 13", doc)

    def test_perturbing_control_fails_validation(self):
        bad = [{"fact": "grasses", "true": "10cm", "q": "?", "steps": [
            {"severity": 0, "replace": [("exceed 10cm", "exceed 15cm")], "target_string": "10cm", "ratio": 1}]}]
        with mock.patch("harness.PERTURBATION_LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("must not perturb" in p for p in problems))

    def test_control_target_string_absent_fails_validation(self):
        bad = [{"fact": "grasses", "true": "10cm", "q": "?", "steps": [
            {"severity": 0, "replace": [], "target_string": "zzqx", "ratio": 1}]}]
        with mock.patch("harness.PERTURBATION_LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("not found in the document" in p for p in problems))


class TestSeverityClassify(unittest.TestCase):
    def test_abstention_detected(self):
        self.assertEqual(classify("NOT IN DOCUMENT", SILENT), "abstained")

    def test_abstention_beats_stance(self):
        self.assertEqual(classify("NOT IN DOCUMENT. That figure would be implausible.", QUESTIONED), "abstained")

    def test_questioned_follows_judge(self):
        self.assertEqual(classify("It says 500 metres, which is implausible.", QUESTIONED), "questioned")

    def test_silent_follows_judge(self):
        self.assertEqual(classify("It says 500 metres.", SILENT), "silent")

    def test_endorsed_follows_judge(self):
        self.assertEqual(classify("It says 500 metres, which is standard.", ENDORSED), "endorsed")

    def test_lexical_caveat_catches_caveat(self):
        self.assertTrue(lexical_caveat("This value seems implausible."))

    def test_lexical_caveat_absent_on_plain_report(self):
        self.assertFalse(lexical_caveat("The maximum height is 30cm."))


class TestAppears(unittest.TestCase):
    def test_no_substring_match(self):
        self.assertFalse(appears("230", "approved in 2023"))

    def test_phrase_token(self):
        self.assertTrue(appears("BAL 19", "rated BAL 19 under AS 3959"))
        self.assertFalse(appears("BAL 19", "see condition 19. Approval to Operate"))

    def test_case_insensitive(self):
        self.assertTrue(appears("not in document", "NOT IN DOCUMENT"))


class TestUnanswerableItems(unittest.TestCase):
    def test_validate_items_clean(self):
        self.assertEqual(validate_items(), [])

    def test_parametric_answers_absent_from_document(self):
        for p in UNANSWERABLE_ITEMS:
            self.assertFalse(appears(p["parametric_answer"], passage), p["item_id"])

    def test_two_probes_per_prior_level(self):
        for lv in PRIOR_STRENGTHS:
            self.assertEqual(sum(1 for p in UNANSWERABLE_ITEMS if p["prior_strength"] == lv), 2)

    def test_item_ids_unique(self):
        item_ids = [p["item_id"] for p in UNANSWERABLE_ITEMS]
        self.assertEqual(len(item_ids), len(set(item_ids)))

    def test_required_fields(self):
        for p in UNANSWERABLE_ITEMS:
            for field in ("item_id", "prior_strength", "proximity", "domain", "parametric_answer", "q"):
                self.assertIn(field, p)
            self.assertIn(p["proximity"], ("near", "far"))

    def test_bare_19_collides_but_phrase_does_not(self):
        self.assertTrue(appears("19", passage))
        self.assertFalse(appears("BAL 19", passage))

    def test_wrong_item_count_fails_validation(self):
        extra = UNANSWERABLE_ITEMS + [{"item_id": "extra", "prior_strength": 6, "proximity": "near",
                                "domain": "x", "parametric_answer": "zzqx", "q": "?"}]
        with mock.patch("harness.UNANSWERABLE_ITEMS", extra):
            self.assertTrue(any("items !=" in p for p in validate_items()))


class TestLegacyGoldGuard(unittest.TestCase):
    def test_legacy_schema_raises_before_judging(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"question": "x", "firmness": "FIRM", "role": "clean-ungrounded anchor",
                        "answer": "y", "human": FAITHFUL}], f)
            path = f.name
        try:
            with self.assertRaises(SystemExit):
                _meta_evaluate(path, "unused.json", "judge", "m", (FAITHFUL, UNGROUNDED),
                         lambda row: self.fail("judge must not be called"), "instruction")
        finally:
            os.unlink(path)


class TestLoadDone(unittest.TestCase):
    def test_counts_by_fields(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for row in [{"model": "m", "instruction": "S", "item_id": "a"},
                        {"model": "m", "instruction": "S", "item_id": "a"},
                        {"model": "m", "instruction": "P", "item_id": "b"}]:
                f.write(json.dumps(row) + "\n")
            path = f.name
        try:
            done = load_done(path, ["model", "instruction", "item_id"])
            self.assertEqual(done[("m", "S", "a")], 2)
            self.assertEqual(done[("m", "P", "b")], 1)
        finally:
            os.unlink(path)

    def test_missing_file_is_empty(self):
        self.assertEqual(load_done("no_such_results_file.jsonl", ["model"]), {})


class TestGoldSchedule(unittest.TestCase):
    def test_twenty_rows_at_reps_2(self):
        self.assertEqual(len(gold_schedule(UNANSWERABLE_ITEMS, 2)), 20)

    def test_both_anchor_classes_present(self):
        roles = {role for _, _, role in gold_schedule(UNANSWERABLE_ITEMS, 2)}
        self.assertIn("clean-ungrounded anchor", roles)
        self.assertIn("clean-faithful anchor", roles)

    def test_strong_priors_soft_weak_priors_strict(self):
        for p, iname, role in gold_schedule(UNANSWERABLE_ITEMS, 2):
            if p["prior_strength"] >= 4:
                self.assertEqual((iname, role), ("WEAK_GROUNDING", "clean-ungrounded anchor"))
            elif p["prior_strength"] <= 2:
                self.assertEqual((iname, role), ("SOURCE_EXCLUSIVE", "clean-faithful anchor"))

    def test_borderline_p3_under_both_instructions(self):
        borderline = [(p["item_id"], iname) for p, iname, role in gold_schedule(UNANSWERABLE_ITEMS, 2)
                      if role == "borderline"]
        self.assertTrue(borderline)
        for item_id in {s for s, _ in borderline}:
            self.assertEqual({i for s, i in borderline if s == item_id}, {"SOURCE_EXCLUSIVE", "WEAK_GROUNDING"})


class TestTradeoffRows(unittest.TestCase):
    def _caveat(self, instr, level, label):
        return {"model": MODELS[0][0], "instruction": instr, "severity": level, "label": label}

    def _ungrounded(self, instr, prior, label):
        return {"model": MODELS[0][0], "instruction": instr, "prior_strength": prior, "label": label}

    def test_reports_each_severity_separately(self):
        caveat = [self._caveat("SOURCE_EXCLUSIVE", 5, QUESTIONED), self._caveat("SOURCE_EXCLUSIVE", 4, SILENT),
                self._caveat("SOURCE_EXCLUSIVE", 1, QUESTIONED)]
        ungrounded = [self._ungrounded("SOURCE_EXCLUSIVE", 5, UNGROUNDED), self._ungrounded("SOURCE_EXCLUSIVE", 3, FAITHFUL)]
        entries = {e["severity"]: e for e in tradeoff_rows(caveat, ungrounded) if e["instruction"] == "SOURCE_EXCLUSIVE"}
        self.assertEqual(set(entries), {1, 3, 4, 5})
        self.assertEqual(entries[5]["caveat_n"], 1)
        self.assertAlmostEqual(entries[5]["caveat_rate"], 1.0)
        self.assertEqual(entries[5]["abstention_n"], 1)
        self.assertAlmostEqual(entries[5]["faithful_rate"], 0.0)
        self.assertEqual(entries[4]["caveat_n"], 1)
        self.assertAlmostEqual(entries[4]["caveat_rate"], 0.0)
        self.assertIsNone(entries[4]["faithful_rate"])
        self.assertEqual(entries[3]["abstention_n"], 1)
        self.assertAlmostEqual(entries[3]["faithful_rate"], 1.0)
        self.assertIsNone(entries[3]["caveat_rate"])

    def test_one_side_missing_reports_other(self):
        entry = tradeoff_rows([self._caveat("FLAG_INVITING", 5, QUESTIONED)], [])[0]
        self.assertEqual(entry["severity"], 5)
        self.assertIsNone(entry["faithful_rate"])
        self.assertEqual(entry["caveat_n"], 1)
        self.assertAlmostEqual(entry["caveat_rate"], 1.0)

    def test_both_empty(self):
        self.assertEqual(tradeoff_rows([], []), [])


if __name__ == "__main__":
    unittest.main()
