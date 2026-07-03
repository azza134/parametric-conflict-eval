# Tests to verify the functions are working as intended. No API key required.
import json
import math
import os
import tempfile
import unittest
from unittest import mock

from harness import (perturb, wilsons, with_retry, PERMISSIVE, INSTRUCTIONS,
                     LADDERS, LEVELS, validate_ladders, total_steps, classify, lexical_flag,
                     LEAK_PROBES, LEAK_INSTRUCTIONS, validate_probes, appears, load_done,
                     balance_rows, passage, PRIORS, step_doc)
from judge import (cohens_kappa, is_faithful, FAITHFUL, LEAK,
                   judge_gate, anchor_disagreements, GATE_PASS, GATE_FAIL, KAPPA_THRESHOLD,
                   is_flagged, FLAGGED, NOT_FLAGGED, build_flag_prompt, FLAG_SCHEMA,
                   gold_schedule, _certify, CONTROL_PROBES)


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


class TestFlagLabels(unittest.TestCase):
    def test_is_flagged_maps_words_to_bools(self):
        self.assertTrue(is_flagged(FLAGGED))
        self.assertFalse(is_flagged(NOT_FLAGGED))


class TestFlagGate(unittest.TestCase):
    def _row(self, role, human, judge):
        return {"role": role, "human": human, "judge": judge}

    def _agree_anchor(self):
        return self._row("absurd anchor", FLAGGED, FLAGGED)

    def test_flag_anchor_agreement_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], 0.9)
        self.assertEqual(verdict, GATE_PASS)

    def test_flagged_anchor_miss_fails_despite_high_kappa(self):
        verdict, _ = judge_gate([self._row("absurd anchor", FLAGGED, NOT_FLAGGED)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)

    def test_notflagged_anchor_miss_fails(self):
        verdict, _ = judge_gate([self._row("absurd anchor", NOT_FLAGGED, FLAGGED)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)

    def test_subtle_borderline_disagreement_passes(self):
        rows = [self._agree_anchor(), self._row("subtle borderline", NOT_FLAGGED, FLAGGED)]
        verdict, _ = judge_gate(rows, 0.9)
        self.assertEqual(verdict, GATE_PASS)

    def test_no_anchor_rows_fails(self):
        verdict, _ = judge_gate([self._row("subtle borderline", FLAGGED, FLAGGED)], 0.95)
        self.assertEqual(verdict, GATE_FAIL)

    def test_anchor_disagreements_picks_absurd_anchor(self):
        rows = [self._row("absurd anchor", FLAGGED, NOT_FLAGGED),
                self._row("subtle borderline", FLAGGED, NOT_FLAGGED)]
        bad = anchor_disagreements(rows)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["role"], "absurd anchor")


class TestFlagPrompt(unittest.TestCase):
    def test_prompt_contains_its_two_args(self):
        prompt = build_flag_prompt("How tall is X?", "It is 500 metres.")
        self.assertIn("How tall is X?", prompt)
        self.assertIn("It is 500 metres.", prompt)

    def test_prompt_takes_no_document(self):
        with self.assertRaises(TypeError):
            build_flag_prompt("q", "a", "the correct value is 10cm")


class TestFlagSchema(unittest.TestCase):
    def test_schema_shape(self):
        self.assertEqual(FLAG_SCHEMA["required"], ["flagged", "reason"])
        self.assertIs(FLAG_SCHEMA["additionalProperties"], False)
        self.assertEqual(FLAG_SCHEMA["properties"]["flagged"]["type"], "boolean")


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
        with mock.patch("harness.time.sleep"):
            self.assertEqual(with_retry(fn, attempts=5), "ok")
        self.assertEqual(state["n"], 3)


class TestInstructions(unittest.TestCase):
    def test_names_are_strict_then_permissive(self):
        self.assertEqual([name for name, _ in INSTRUCTIONS], ["STRICT", "PERMISSIVE"])

    def test_permissive_invites_flagging(self):
        self.assertIn("flag", PERMISSIVE.lower())


class TestLadders(unittest.TestCase):
    def test_all_perturbations_apply_to_document(self):
        self.assertEqual(validate_ladders(), [])

    def test_every_fact_ladders_0_to_5(self):
        for fact in LADDERS:
            self.assertEqual([s["level"] for s in fact["steps"]], LEVELS)

    def test_thirty_six_steps_total(self):
        self.assertEqual(total_steps(), 36)

    def test_ratio_increases_where_present(self):
        for fact in LADDERS:
            ratios = [s["ratio"] for s in fact["steps"] if s["ratio"] is not None]
            self.assertEqual(ratios, sorted(ratios))

    def test_saturday_hours_is_non_ratio(self):
        sat = next(f for f in LADDERS if f["fact"] == "saturday_hours")
        self.assertTrue(all(s["ratio"] is None for s in sat["steps"]))

    def test_noop_perturbation_reported_via_exception_alone(self):
        with mock.patch("harness.perturb", side_effect=AssertionError("no change in passage detected")):
            problems = validate_ladders()
        self.assertTrue(any("no change in passage detected" in p for p in problems))
        self.assertEqual(len(problems), total_steps() - len(LADDERS))


class TestControlRung(unittest.TestCase):
    def test_every_fact_leads_with_an_unperturbed_control(self):
        for fact in LADDERS:
            first = fact["steps"][0]
            self.assertEqual(first["level"], 0, fact["fact"])
            self.assertEqual(first["replace"], [], fact["fact"])

    def test_control_tokens_present_in_document(self):
        for fact in LADDERS:
            self.assertTrue(appears(fact["steps"][0]["token"], passage), fact["fact"])

    def test_step_doc_control_is_the_real_passage(self):
        self.assertEqual(step_doc({"replace": []}), passage)

    def test_step_doc_applies_perturbation(self):
        doc = step_doc({"replace": [("every 20", "every 13")]})
        self.assertNotEqual(doc, passage)
        self.assertIn("every 13", doc)

    def test_perturbing_control_fails_validation(self):
        bad = [{"fact": "grasses", "true": "10cm", "q": "?", "steps": [
            {"level": 0, "replace": [("exceed 10cm", "exceed 15cm")], "token": "10cm", "ratio": 1}]}]
        with mock.patch("harness.LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("must not perturb" in p for p in problems))

    def test_control_token_absent_fails_validation(self):
        bad = [{"fact": "grasses", "true": "10cm", "q": "?", "steps": [
            {"level": 0, "replace": [], "token": "zzqx", "ratio": 1}]}]
        with mock.patch("harness.LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("not found in the document" in p for p in problems))


class TestControlProbes(unittest.TestCase):
    def test_controls_are_gating_anchors(self):
        for p in CONTROL_PROBES:
            self.assertIn("anchor", p["role"])

    def test_controls_do_not_perturb(self):
        for p in CONTROL_PROBES:
            self.assertEqual(p["replace"], [])

    def test_control_questions_come_from_the_ladders(self):
        qs = {f["q"] for f in LADDERS}
        for p in CONTROL_PROBES:
            self.assertIn(p["q"], qs)


class TestMagnitudeClassify(unittest.TestCase):
    def test_refusal_detected(self):
        self.assertEqual(classify("NOT IN DOCUMENT", False), "refused")

    def test_flagged_follows_judge(self):
        self.assertEqual(classify("It says 500 metres.", True), "flagged")

    def test_reported_when_judge_does_not_flag(self):
        self.assertEqual(classify("It says 500 metres.", False), "reported")

    def test_lexical_flag_catches_caveat(self):
        self.assertTrue(lexical_flag("This value seems implausible."))

    def test_lexical_flag_absent_on_plain_report(self):
        self.assertFalse(lexical_flag("The maximum height is 30cm."))


class TestAppears(unittest.TestCase):
    def test_no_substring_match(self):
        self.assertFalse(appears("230", "approved in 2023"))

    def test_phrase_token(self):
        self.assertTrue(appears("BAL 19", "rated BAL 19 under AS 3959"))
        self.assertFalse(appears("BAL 19", "see condition 19. Approval to Operate"))

    def test_case_insensitive(self):
        self.assertTrue(appears("not in document", "NOT IN DOCUMENT"))


class TestLeakProbes(unittest.TestCase):
    def test_validate_probes_clean(self):
        self.assertEqual(validate_probes(), [])

    def test_avoid_tokens_absent_from_document(self):
        for p in LEAK_PROBES:
            self.assertFalse(appears(p["avoid"], passage), p["slug"])

    def test_two_probes_per_prior_level(self):
        for lv in PRIORS:
            self.assertEqual(sum(1 for p in LEAK_PROBES if p["prior"] == lv), 2)

    def test_slugs_unique(self):
        slugs = [p["slug"] for p in LEAK_PROBES]
        self.assertEqual(len(slugs), len(set(slugs)))

    def test_required_fields(self):
        for p in LEAK_PROBES:
            for field in ("slug", "prior", "proximity", "domain", "avoid", "q"):
                self.assertIn(field, p)
            self.assertIn(p["proximity"], ("near", "far"))

    def test_bare_19_collides_but_phrase_does_not(self):
        self.assertTrue(appears("19", passage))
        self.assertFalse(appears("BAL 19", passage))

    def test_wrong_probe_count_fails_validation(self):
        extra = LEAK_PROBES + [{"slug": "extra", "prior": 6, "proximity": "near",
                                "domain": "x", "avoid": "zzqx", "q": "?"}]
        with mock.patch("harness.LEAK_PROBES", extra):
            self.assertTrue(any("probes !=" in p for p in validate_probes()))


class TestLegacyGoldGuard(unittest.TestCase):
    def test_legacy_schema_raises_before_judging(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"question": "x", "firmness": "FIRM", "role": "clean-leak anchor",
                        "answer": "y", "human": FAITHFUL}], f)
            path = f.name
        try:
            with self.assertRaises(SystemExit):
                _certify(path, "unused.json", "judge", "m", FAITHFUL, LEAK, is_faithful,
                         lambda row: self.fail("judge must not be called"), "instruction")
        finally:
            os.unlink(path)


class TestLeakInstructions(unittest.TestCase):
    def test_names_and_order(self):
        self.assertEqual([n for n, _ in LEAK_INSTRUCTIONS], ["STRICT", "PERMISSIVE", "SOFT"])

    def test_three_distinct_instructions(self):
        self.assertEqual(len({t for _, t in LEAK_INSTRUCTIONS}), 3)


class TestLoadDone(unittest.TestCase):
    def test_counts_by_fields(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for row in [{"model": "m", "instruction": "S", "slug": "a"},
                        {"model": "m", "instruction": "S", "slug": "a"},
                        {"model": "m", "instruction": "P", "slug": "b"}]:
                f.write(json.dumps(row) + "\n")
            path = f.name
        try:
            done = load_done(path, ["model", "instruction", "slug"])
            self.assertEqual(done[("m", "S", "a")], 2)
            self.assertEqual(done[("m", "P", "b")], 1)
        finally:
            os.unlink(path)

    def test_missing_file_is_empty(self):
        self.assertEqual(load_done("no_such_results_file.jsonl", ["model"]), {})


class TestGoldSchedule(unittest.TestCase):
    def test_twenty_rows_at_reps_2(self):
        self.assertEqual(len(gold_schedule(LEAK_PROBES, 2)), 20)

    def test_both_anchor_classes_present(self):
        roles = {role for _, _, role in gold_schedule(LEAK_PROBES, 2)}
        self.assertIn("clean-leak anchor", roles)
        self.assertIn("clean-faithful anchor", roles)

    def test_strong_priors_soft_weak_priors_strict(self):
        for p, iname, role in gold_schedule(LEAK_PROBES, 2):
            if p["prior"] >= 4:
                self.assertEqual((iname, role), ("SOFT", "clean-leak anchor"))
            elif p["prior"] <= 2:
                self.assertEqual((iname, role), ("STRICT", "clean-faithful anchor"))

    def test_borderline_p3_under_both_instructions(self):
        borderline = [(p["slug"], iname) for p, iname, role in gold_schedule(LEAK_PROBES, 2)
                      if role == "borderline"]
        self.assertTrue(borderline)
        for slug in {s for s, _ in borderline}:
            self.assertEqual({i for s, i in borderline if s == slug}, {"STRICT", "SOFT"})


class TestBalanceRows(unittest.TestCase):
    def _flag(self, instr, level, label):
        return {"model": "claude-sonnet-5", "instruction": instr, "level": level, "label": label}

    def _leak(self, instr, prior, label):
        return {"model": "claude-sonnet-5", "instruction": instr, "prior": prior, "label": label}

    def test_reports_each_level_separately(self):
        flag = [self._flag("STRICT", 5, "flagged"), self._flag("STRICT", 4, "reported"),
                self._flag("STRICT", 1, "flagged")]
        leak = [self._leak("STRICT", 5, LEAK), self._leak("STRICT", 3, LEAK)]
        entries = {e["level"]: e for e in balance_rows(flag, leak) if e["instruction"] == "STRICT"}
        self.assertEqual(set(entries), {1, 3, 4, 5})
        self.assertEqual(entries[5]["flag_n"], 1)
        self.assertAlmostEqual(entries[5]["flag_rate"], 1.0)
        self.assertEqual(entries[5]["leak_n"], 1)
        self.assertAlmostEqual(entries[5]["leak_rate"], 1.0)
        self.assertEqual(entries[4]["flag_n"], 1)
        self.assertAlmostEqual(entries[4]["flag_rate"], 0.0)
        self.assertIsNone(entries[4]["leak_rate"])
        self.assertEqual(entries[3]["leak_n"], 1)
        self.assertIsNone(entries[3]["flag_rate"])

    def test_one_side_missing_reports_other(self):
        entry = balance_rows([self._flag("PERMISSIVE", 5, "flagged")], [])[0]
        self.assertEqual(entry["level"], 5)
        self.assertIsNone(entry["leak_rate"])
        self.assertEqual(entry["flag_n"], 1)
        self.assertAlmostEqual(entry["flag_rate"], 1.0)

    def test_both_empty(self):
        self.assertEqual(balance_rows([], []), [])


if __name__ == "__main__":
    unittest.main()
