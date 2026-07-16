# Tests to verify the functions are working as intended. No API key required.
import json
import math
import os
import tempfile
import types
import unittest
from unittest import mock

import config

from config import (perturb, with_retry, DOCUMENTS, doc_text, openai_reasoning_kwargs, MODELS, FLAG_INVITING, SOURCE_EXCLUSIVE,
                    SOURCE_EXCLUSIVE_FLAG_INVITING, SYSTEM_INSTRUCTIONS,
                    appears, passage, step_doc, build_batch_message_params)
from harness import (wilson_interval, PERTURBATION_LADDERS, SEVERITIES, validate_ladders,
                     ABSENCE_PATCHES, absence_doc, validate_absence, _absence_row,
                     encode_absence_custom_id, decode_absence_custom_id, absence_wave_plan,
                     total_steps, total_cells, derive_label, lexical_caveat, UNANSWERABLE_ITEMS, validate_items,
                     load_done, cluster_icc, vector_cells,
                     probe_targets, _probe_row,
                     encode_caveat_custom_id, decode_caveat_custom_id,
                     encode_abstention_custom_id, decode_abstention_custom_id,
                     caveat_wave_plan, abstention_wave_plan, concurrent_map, pilot_selection,
                     _run_anthropic_wave, _chunked_judge_sink, build_manifest,
                     SweepSpec, CAVEAT_SWEEP, ABSTENTION_SWEEP, ABSENCE_SWEEP, _sweep_wave_plan, _run_sweep,
                     sign_test, bootstrap_ci, unit_counts, unit_rate_map, factorial_effects, _situated_faithfulness,
                     _openai_batch_chunks)
from judge import (cohens_kappa, FAITHFUL, UNGROUNDED,
                   judge_gate, anchor_disagreements, GATE_PASS, GATE_FAIL, KAPPA_THRESHOLD,
                   QUESTIONED, SILENT, ENDORSED, DECLINED, CAVEAT_LABELS, CAVEAT_SCHEMA, build_caveat_prompt,
                   CORROBORATION_LABELS,
                   gold_schedule, expand_schedule, _meta_evaluate)


class TestWilsonInterval(unittest.TestCase):
    def test_point_estimate(self):
        p, low, high = wilson_interval(3, 5)
        self.assertAlmostEqual(p, 0.6)
        self.assertTrue(0.0 <= low <= p <= high <= 1.0)

    def test_extreme_is_clamped_and_honest(self):
        p, low, high = wilson_interval(5, 5)
        self.assertEqual(p, 1.0)
        self.assertEqual(high, 1.0)   
        self.assertLess(low, 1.0)     


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
    def _row(self, role, human, judge):
        return {"role": role, "human": human, "judge": judge}

    def _agree_anchor(self):  
        return self._row("clean-ungrounded anchor", FAITHFUL, FAITHFUL)

    def test_clean_anchors_high_kappa_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], 0.95)
        self.assertEqual(verdict, GATE_PASS)

    def test_anchor_disagreement_fails_despite_high_kappa(self):
        verdict, reasons = judge_gate([self._row("clean-ungrounded anchor", FAITHFUL, UNGROUNDED)], 1.0)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("anchor" in r.lower() for r in reasons)) # ensures the reason attached in this gate fail scenario involves anchor

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
        self.assertTrue(any("WARNING" in r for r in reasons)) # ensures even nan kappa passes with a warning attached

    def test_nan_kappa_anchor_disagreement_still_fails(self):
        verdict, _ = judge_gate([self._row("clean-faithful anchor", FAITHFUL, UNGROUNDED)], float("nan"))
        self.assertEqual(verdict, GATE_FAIL)

    def test_kappa_exactly_threshold_passes(self):
        verdict, _ = judge_gate([self._agree_anchor()], KAPPA_THRESHOLD) 
        self.assertEqual(verdict, GATE_PASS) 

    def test_anchor_matching_is_substring(self):
        rows = [self._row("clean-ungrounded anchor", FAITHFUL, UNGROUNDED),
                self._row("clean-faithful anchor", FAITHFUL, UNGROUNDED),
                self._row("borderline", FAITHFUL, UNGROUNDED)]
        bad = anchor_disagreements(rows)
        self.assertEqual(len(bad), 2)  
        self.assertTrue(all("anchor" in r["role"] for r in bad))

    def test_no_anchor_rows_fails(self):
        verdict, reasons = judge_gate([self._row("borderline", FAITHFUL, FAITHFUL)], 0.95)
        self.assertEqual(verdict, GATE_FAIL)
        self.assertTrue(any("anchor" in r.lower() for r in reasons))


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


class TestInstructions(unittest.TestCase):
    def test_instruction_names_and_order(self):
        self.assertEqual([name for name, _ in SYSTEM_INSTRUCTIONS],
                         ["SOURCE_EXCLUSIVE", "FLAG_INVITING", "WEAK_GROUNDING", "SOURCE_EXCLUSIVE_FLAG_INVITING",
                          "SELECTIVE_AUDIT"])

    def test_five_distinct_instructions(self):
        self.assertEqual(len({t for _, t in SYSTEM_INSTRUCTIONS}), 5)

    def test_selective_audit_gates_by_evidence_state(self):
        text = dict(SYSTEM_INSTRUCTIONS)["SELECTIVE_AUDIT"]
        self.assertIn("NOT IN DOCUMENT", text)
        self.assertIn("flag the conflict", text)
        self.assertIn("do not replace", text)

    def test_max_custom_id_length_within_anthropic_ceiling(self):
        longest_instr = max((name for name, _ in SYSTEM_INSTRUCTIONS), key=len)
        longest_fact = max((f["fact"] for f in PERTURBATION_LADDERS), key=len)
        longest_item = max((p["item_id"] for p in UNANSWERABLE_ITEMS), key=len)
        worst = max(len(encode_caveat_custom_id(longest_fact, 5, longest_instr, 2)),
                    len(encode_abstention_custom_id(longest_item, longest_instr, 2)),
                    len(encode_absence_custom_id(longest_fact, longest_instr, 2)))
        self.assertLessEqual(worst, 64)

    def test_permissive_invites_flagging(self): # lexical check that 'flag' is in 'flag inviting' instruction
        self.assertIn("flag", FLAG_INVITING.lower())

    def test_composed_arm_reuses_source_exclusive_verbatim(self):
        self.assertTrue(SOURCE_EXCLUSIVE_FLAG_INVITING.startswith(SOURCE_EXCLUSIVE))
        self.assertIn("flag your concern", SOURCE_EXCLUSIVE_FLAG_INVITING)

    def test_no_hyphens_in_instruction_names(self): # hyphens are a delimiter in harness.py that constrains what can be named
        for name, _ in SYSTEM_INSTRUCTIONS:
            self.assertNotIn("-", name)

    def test_no_hyphens_in_fact_names(self):
        for fact in PERTURBATION_LADDERS:
            self.assertNotIn("-", fact["fact"])


class TestClusterIcc(unittest.TestCase): # anova splits variance, icc converts the split to compute redundancy, n_eff uses icc discounts sample size which goes to wilsons
    def test_all_or_nothing_clusters(self):
        p, icc, n_eff = cluster_icc([(0, 8), (0, 8), (0, 8), (8, 8), (8, 8), (8, 8)])
        self.assertEqual(p, 0.5)
        self.assertEqual(icc, 1.0) # one-way random-effects, where clusters are random and repeats are nested inside
        self.assertAlmostEqual(n_eff, 6.0) 

    def test_degenerate_cell_has_no_icc(self): # no variation 
        self.assertEqual(cluster_icc([(0, 8)] * 6), (0.0, None, None))
        self.assertEqual(cluster_icc([(8, 8)] * 6), (1.0, None, None))

    def test_uncorrelated_reps_keep_full_n(self):
        p, icc, n_eff = cluster_icc([(1, 8), (0, 8), (0, 8), (0, 8), (1, 8), (0, 8)])
        self.assertEqual(icc, 0.0)
        self.assertAlmostEqual(n_eff, 48.0)


class TestPriorProbe(unittest.TestCase):
    def test_forty_targets_covering_all_facts_and_items(self):
        targets = probe_targets()
        self.assertEqual(len(targets), len(PERTURBATION_LADDERS) + len(UNANSWERABLE_ITEMS))
        self.assertEqual({t["kind"] for t in targets}, {"fact", "item"})
        for t in targets:
            self.assertIn(t["doc"], DOCUMENTS)
            self.assertTrue(t["q"] and t["expected"]) # ensures q and expected are not empty

    def test_item_targets_use_prior_strength_as_rating(self):
        t = next(t for t in probe_targets() if t["name"] == "water_boil")
        self.assertEqual(t["prior_rating"], 5)

    def test_probe_row_lexical_flags(self):
        t = {"kind": "fact", "name": "x", "doc": "consent", "q": "?", "expected": "20", "accepted": ["20"], "prior_rating": 3}
        row = _probe_row("m", "openai", t, "The value is 20 persons.")
        self.assertTrue(row["reports_expected"])
        self.assertFalse(row["says_dont_know"])
        row = _probe_row("m", "openai", t, "I do not know.")
        self.assertFalse(row["reports_expected"])
        self.assertTrue(row["says_dont_know"])

    def test_probe_row_stamps_truncated(self):
        t = {"kind": "fact", "name": "x", "doc": "consent", "q": "?", "expected": "20", "accepted": ["20"], "prior_rating": 3}
        self.assertFalse(_probe_row("m", "openai", t, "20")["truncated"])
        self.assertTrue(_probe_row("m", "openai", t, "20", "snap", True)["truncated"])


class TestLadders(unittest.TestCase):
    def test_all_perturbations_apply_to_document(self):
        self.assertEqual(validate_ladders(), [])

    def test_every_fact_ladders_0_to_5(self):
        for fact in PERTURBATION_LADDERS:
            self.assertEqual([s["severity"] for s in fact["steps"]], SEVERITIES)

    def test_one_hundred_forty_four_steps_total(self):
        self.assertEqual(total_steps(), 144)

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
            self.assertTrue(appears(fact["steps"][0]["target_string"], doc_text(fact["doc"])), fact["fact"])

    def test_step_doc_control_is_the_real_passage(self):
        self.assertEqual(step_doc({"doc": "consent"}, {"replace": []}), passage)

    def test_step_doc_applies_perturbation(self):
        doc = step_doc({"doc": "consent"}, {"replace": [("every 20", "every 13")]})
        self.assertNotEqual(doc, passage)
        self.assertIn("every 13", doc)

    def test_perturbing_control_fails_validation(self):
        bad = [{"fact": "grasses", "doc": "consent", "true": "10cm", "q": "?", "steps": [
            {"severity": 0, "replace": [("exceed 10cm", "exceed 15cm")], "target_string": "10cm", "ratio": 1}]}]
        with mock.patch("harness.PERTURBATION_LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("must not perturb" in p for p in problems))

    def test_control_target_string_absent_fails_validation(self):
        bad = [{"fact": "grasses", "doc": "consent", "true": "10cm", "q": "?", "steps": [
            {"severity": 0, "replace": [], "target_string": "zzqx", "ratio": 1}]}]
        with mock.patch("harness.PERTURBATION_LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("not found in the document" in p for p in problems))


class TestMultiDocument(unittest.TestCase):
    def test_registry_loads_three_documents(self):
        self.assertEqual(set(DOCUMENTS), {"consent", "epl", "liquor"})
        for name in DOCUMENTS:
            self.assertGreater(len(doc_text(name).split()), 1000, name)

    def test_every_fact_and_item_declares_a_known_doc(self):
        for fact in PERTURBATION_LADDERS:
            self.assertIn(fact.get("doc"), DOCUMENTS, fact["fact"])
        for p in UNANSWERABLE_ITEMS:
            self.assertIn(p.get("doc"), DOCUMENTS, p["item_id"])

    def test_step_doc_uses_the_facts_own_document(self):
        base = step_doc({"doc": "liquor"}, {"replace": []})
        self.assertEqual(base, doc_text("liquor"))
        self.assertNotEqual(base, passage)

    def test_fact_with_unknown_doc_fails_validation(self):
        bad = [{"fact": "grasses", "doc": "zzqx", "true": "10cm", "q": "?", "steps": [
            {"severity": 0, "replace": [], "target_string": "10cm", "ratio": 1}]}]
        with mock.patch("harness.PERTURBATION_LADDERS", bad):
            problems = validate_ladders()
        self.assertTrue(any("not in DOCUMENTS" in p for p in problems))

    def test_item_with_missing_doc_fails_validation(self):
        bad = [dict(p) for p in UNANSWERABLE_ITEMS]
        del bad[0]["doc"]
        with mock.patch("harness.UNANSWERABLE_ITEMS", bad):
            problems = validate_items()
        self.assertTrue(any("not in DOCUMENTS" in p for p in problems))


class TestSeverityClassify(unittest.TestCase):
    def test_abstention_detected(self):
        self.assertEqual(derive_label("NOT IN DOCUMENT", SILENT), "abstained")

    def test_abstention_beats_stance(self):
        self.assertEqual(derive_label("NOT IN DOCUMENT. That figure would be implausible.", QUESTIONED), "abstained")

    def test_questioned_follows_judge(self):
        self.assertEqual(derive_label("It says 500 metres, which is implausible.", QUESTIONED), "questioned")

    def test_silent_follows_judge(self):
        self.assertEqual(derive_label("It says 500 metres.", SILENT), "silent")

    def test_endorsed_follows_judge(self):
        self.assertEqual(derive_label("It says 500 metres, which is standard.", ENDORSED), "endorsed")

    def test_lexical_caveat_catches_caveat(self):
        self.assertTrue(lexical_caveat("This value seems implausible."))

    def test_lexical_caveat_absent_on_plain_report(self):
        self.assertFalse(lexical_caveat("The maximum height is 30cm."))


class TestUnanswerableItems(unittest.TestCase):
    def test_validate_items_clean(self):
        self.assertEqual(validate_items(), [])

    def test_parametric_answers_absent_from_document(self):
        for p in UNANSWERABLE_ITEMS:
            self.assertFalse(appears(p["parametric_answer"], doc_text(p["doc"])), p["item_id"])

    def test_twenty_four_items_balanced_bins(self):
        self.assertEqual(len(UNANSWERABLE_ITEMS), 24)

    def test_item_ids_unique(self):
        item_ids = [p["item_id"] for p in UNANSWERABLE_ITEMS]
        self.assertEqual(len(item_ids), len(set(item_ids)))

    def test_required_fields(self):
        for p in UNANSWERABLE_ITEMS:
            for field in ("item_id", "doc", "prior_strength", "proximity", "domain", "parametric_answer", "q"):
                self.assertIn(field, p)
            self.assertIn(p["proximity"], ("near", "far"))

    def test_bare_19_collides_but_phrase_does_not(self):
        self.assertTrue(appears("19", passage))
        self.assertFalse(appears("BAL 19", passage))

    def test_out_of_range_prior_fails_validation(self):
        extra = UNANSWERABLE_ITEMS + [{"item_id": "extra", "doc": "consent", "prior_strength": 6, "proximity": "near",
                                "domain": "x", "parametric_answer": "zzqx", "q": "?"}]
        with mock.patch("harness.UNANSWERABLE_ITEMS", extra):
            self.assertTrue(any("outside 1-5" in p for p in validate_items()))


class TestGoldSchedule(unittest.TestCase):
    def test_forty_eight_rows_at_reps_2(self):
        self.assertEqual(len(gold_schedule(UNANSWERABLE_ITEMS, 2)), 48)

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


class TestMatchedAbsence(unittest.TestCase):
    def test_every_fact_has_a_patch(self):
        self.assertEqual(set(ABSENCE_PATCHES), {f["fact"] for f in PERTURBATION_LADDERS})
        for f in PERTURBATION_LADDERS:
            self.assertIn("absence", f)

    def test_validation_clean_on_real_data(self):
        self.assertEqual(validate_absence(), [])

    def test_deleted_doc_differs_from_base(self):
        for f in PERTURBATION_LADDERS:
            self.assertNotEqual(absence_doc(f), doc_text(f["doc"]))

    def test_codec_roundtrip_and_format(self):
        cid = encode_absence_custom_id("grasses", "SELECTIVE_AUDIT", 2)
        self.assertEqual(cid, "ma-grasses-SELECTIVE_AUDIT-r2")
        self.assertEqual(decode_absence_custom_id(cid),
                         {"fact": "grasses", "instruction": "SELECTIVE_AUDIT", "rep": 2})

    def test_decode_rejects_other_prefixes(self):
        with self.assertRaises(ValueError):
            decode_absence_custom_id(encode_abstention_custom_id("water_boil", "SOURCE_EXCLUSIVE", 0))

    def test_wave_plan_rep0_per_cell_and_resume(self):
        instr = [("SOURCE_EXCLUSIVE", "sys")]
        ladders = [{"fact": "grasses"}, {"fact": "toilets"}]
        w1, w2 = absence_wave_plan({}, 2, "m", instructions=instr, ladders=ladders)
        self.assertEqual(w1, [encode_absence_custom_id("grasses", "SOURCE_EXCLUSIVE", 0),
                              encode_absence_custom_id("toilets", "SOURCE_EXCLUSIVE", 0)])
        self.assertEqual(w2, [encode_absence_custom_id("grasses", "SOURCE_EXCLUSIVE", 1),
                              encode_absence_custom_id("toilets", "SOURCE_EXCLUSIVE", 1)])
        done = {("m", "SOURCE_EXCLUSIVE", "grasses"): 2, ("m", "SOURCE_EXCLUSIVE", "toilets"): 1}
        w1, w2 = absence_wave_plan(done, 2, "m", instructions=instr, ladders=ladders)
        self.assertEqual(w1, [encode_absence_custom_id("toilets", "SOURCE_EXCLUSIVE", 1)])
        self.assertEqual(w2, [])

    def test_wave_plan_real_data_counts(self):
        w1, w2 = absence_wave_plan({}, 1, "claude-sonnet-5")
        self.assertEqual(len(w1), len(PERTURBATION_LADDERS) * len(SYSTEM_INSTRUCTIONS))
        self.assertEqual(w2, [])

    def test_absence_row_judges_against_deleted_doc(self):
        fact = next(f for f in PERTURBATION_LADDERS if f["fact"] == "grasses")
        deleted = absence_doc(fact)
        with mock.patch("harness.abstention_judge", return_value=(True, "abstained", "judge-snap")) as j:
            row = _absence_row("m", "anthropic", "SOURCE_EXCLUSIVE", fact, deleted, "NOT IN DOCUMENT", "snap", 1)
        j.assert_called_once_with(fact["q"], deleted, "NOT IN DOCUMENT")
        self.assertEqual(row["evidence_state"], "absent")
        self.assertEqual(row["label"], FAITHFUL)
        self.assertEqual((row["fact"], row["rep"], row["snapshot"], row["judge_snapshot"]),
                         ("grasses", 1, "snap", "judge-snap"))
        self.assertFalse(row["reports_deleted_value"])
        self.assertTrue(row["verbatim_abstention"])

    def test_absence_row_stamps_truncated(self):
        fact = next(f for f in PERTURBATION_LADDERS if f["fact"] == "grasses")
        with mock.patch("harness.abstention_judge", return_value=(True, "abstained", "judge-snap")):
            self.assertFalse(_absence_row("m", "anthropic", "SOURCE_EXCLUSIVE", fact, "DOC", "a", "snap", 1)["truncated"])
            self.assertTrue(_absence_row("m", "anthropic", "SOURCE_EXCLUSIVE", fact, "DOC", "a", "snap", 1, True)["truncated"])


class TestSignTest(unittest.TestCase):
    def test_all_positive(self):
        p, pos, n = sign_test([0.1, 0.2, 0.3])
        self.assertEqual((pos, n), (3, 3))
        self.assertAlmostEqual(p, 0.25)

    def test_split_is_uninformative(self):
        p, pos, n = sign_test([0.5, -0.5])
        self.assertEqual((pos, n), (1, 2))
        self.assertEqual(p, 1.0)

    def test_zeros_excluded(self):
        p, pos, n = sign_test([0.0, 0.0, 0.4])
        self.assertEqual((pos, n), (1, 1))

    def test_all_zero(self):
        self.assertEqual(sign_test([0.0, 0.0]), (1.0, 0, 0))


class TestBootstrapCI(unittest.TestCase):
    def test_constant_values_give_point_interval(self):
        lo, hi = bootstrap_ci([0.4] * 10, iters=200)
        self.assertAlmostEqual(lo, 0.4)
        self.assertAlmostEqual(hi, 0.4)

    def test_deterministic_and_ordered(self):
        vals = [0.0, 0.2, 0.5, 0.9, 1.0]
        a = bootstrap_ci(vals, iters=500)
        b = bootstrap_ci(vals, iters=500)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])

    def test_interval_within_value_range(self):
        lo, hi = bootstrap_ci([0.1, 0.9], iters=500)
        self.assertGreaterEqual(lo, 0.1)
        self.assertLessEqual(hi, 0.9)


class TestFactorialEffects(unittest.TestCase):
    def rates(self, wg, fi, se, sefi):
        return {"WEAK_GROUNDING": {"f": wg}, "FLAG_INVITING": {"f": fi},
                "SOURCE_EXCLUSIVE": {"f": se}, "SOURCE_EXCLUSIVE_FLAG_INVITING": {"f": sefi}}

    def test_pure_invitation_effect(self):
        effects, usable = factorial_effects(self.rates(0.0, 1.0, 0.0, 1.0), ["f"])
        self.assertEqual(usable, ["f"])
        self.assertEqual(effects["SE_main"], [0.0])
        self.assertEqual(effects["FI_main"], [1.0])
        self.assertEqual(effects["interaction"], [0.0])

    def test_exclusivity_cancels_invitation(self):
        effects, _ = factorial_effects(self.rates(0.0, 1.0, 0.0, 0.0), ["f"])
        self.assertEqual(effects["SE_main"], [-0.5])
        self.assertEqual(effects["FI_main"], [0.5])
        self.assertEqual(effects["interaction"], [-1.0])

    def test_missing_arm_drops_unit(self):
        rates = self.rates(0.0, 1.0, 0.0, 1.0)
        rates["SOURCE_EXCLUSIVE"]["f"] = None
        effects, usable = factorial_effects(rates, ["f"])
        self.assertEqual(usable, [])
        self.assertEqual(effects["FI_main"], [])


class TestSituatedFaithfulness(unittest.TestCase):
    def cav(self, fact, sev, stance, n=3):
        return [{"model": "m", "instruction": "i", "fact": fact, "severity": sev, "stance": stance}] * n

    def ab(self, fact, faithful, n=3):
        return [{"model": "m", "instruction": "i", "fact": fact, "faithful": faithful}] * n

    def test_all_three_states_required(self):
        cav_rows = self.cav("a", 0, "silent") + self.cav("a", 3, "questioned") + self.cav("a", 5, "questioned")
        self.assertEqual(_situated_faithfulness(cav_rows, self.ab("a", True), "m", "i"), (1, 1))
        self.assertEqual(_situated_faithfulness(cav_rows, self.ab("a", False), "m", "i"), (0, 1))

    def test_s0_flag_fails_accept(self):
        cav_rows = self.cav("a", 0, "questioned") + self.cav("a", 3, "questioned") + self.cav("a", 5, "questioned")
        self.assertEqual(_situated_faithfulness(cav_rows, self.ab("a", True), "m", "i"), (0, 1))

    def test_majority_not_unanimity(self):
        cav_rows = (self.cav("a", 0, "silent") + self.cav("a", 3, "questioned", 2) + self.cav("a", 3, "silent", 1)
                    + self.cav("a", 4, "questioned", 2) + self.cav("a", 4, "endorsed", 1)
                    + self.cav("a", 5, "questioned"))
        self.assertEqual(_situated_faithfulness(cav_rows, self.ab("a", True), "m", "i"), (1, 1))


class TestPerturb(unittest.TestCase):
    def test_replaces(self):
        self.assertEqual(perturb("every 20 persons", [("every 20", "every 13")]), "every 13 persons") # when 20 is replaced with 13 does it equal argument 3

    def test_raises_on_noop(self):
        with self.assertRaises(AssertionError):
            perturb("nothing to change here", [("absent token", "x")]) # argument 1 is a phrase that doesn't exist in passage, thus nothing to replace, raising the assertion error for the perturb function

    def test_raises_when_one_of_several_replacements_misses(self):
        with self.assertRaises(AssertionError):
            perturb("every 20 persons", [("every 20", "every 25"), ("part of 20 persons", "part of 25 persons")])


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
        self.assertEqual(CAVEAT_SCHEMA["required"], ["stance", "corroboration", "reason"])
        self.assertIs(CAVEAT_SCHEMA["additionalProperties"], False)
        self.assertEqual(CAVEAT_SCHEMA["properties"]["stance"]["enum"], list(CAVEAT_LABELS))
        self.assertEqual(CAVEAT_SCHEMA["properties"]["corroboration"]["enum"], list(CORROBORATION_LABELS))

    def test_labels_are_the_four_stances(self):
        self.assertEqual(CAVEAT_LABELS, (QUESTIONED, SILENT, ENDORSED, DECLINED))


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


class TestEffortConvention(unittest.TestCase):
    def test_v1_gpt54_candidates_stay_pinned_low(self):
        self.assertEqual(openai_reasoning_kwargs("gpt-5.4-nano"), {"reasoning": {"effort": "low"}})
        self.assertEqual(openai_reasoning_kwargs("gpt-5.4-mini"), {"reasoning": {"effort": "low"}})

    def test_new_models_run_at_vendor_default(self):
        self.assertEqual(openai_reasoning_kwargs("gpt-5.6-terra"), {})
        self.assertEqual(openai_reasoning_kwargs("gpt-4o-mini"), {})


class TestVectorCells(unittest.TestCase):
    def test_groups_by_cell_and_unit(self):
        rows = [
            {"model": "m", "instruction": "i", "severity": 1, "fact": "a", "label": "questioned"},
            {"model": "m", "instruction": "i", "severity": 1, "fact": "a", "label": "silent"},
            {"model": "m", "instruction": "i", "severity": 1, "fact": "b", "label": "questioned"},
            {"model": "m", "instruction": "i", "severity": 2, "fact": "a", "label": "questioned"},
        ]
        cells = vector_cells(rows, "fact", "severity", "questioned")
        self.assertEqual(cells[("m", "i", 1)], {"a": (1, 2), "b": (1, 1)})
        self.assertEqual(cells[("m", "i", 2)], {"a": (1, 1)})


class TestConcurrentMap(unittest.TestCase):
    def test_preserves_order(self):
        items = list(range(50))
        self.assertEqual(concurrent_map(lambda x: x * x, items, workers=8), [x * x for x in items])

    def test_serial_fallback_matches(self):
        self.assertEqual(concurrent_map(str, [1, 2, 3], workers=1), ["1", "2", "3"])

    def test_empty_and_singleton(self):
        self.assertEqual(concurrent_map(lambda x: x, [], workers=8), [])
        self.assertEqual(concurrent_map(lambda x: x + 1, [41], workers=8), [42])

    def test_propagates_exception(self):
        def boom(x):
            if x == 3:
                raise ValueError("boom")
            return x
        with self.assertRaises(ValueError):
            concurrent_map(boom, [1, 2, 3, 4], workers=4)


class TestAppears(unittest.TestCase):
    def test_no_substring_match(self):
        self.assertFalse(appears("230", "approved in 2023"))

    def test_phrase_token(self):
        self.assertTrue(appears("BAL 19", "rated BAL 19 under AS 3959"))
        self.assertFalse(appears("BAL 19", "see condition 19. Approval to Operate"))

    def test_case_insensitive(self):
        self.assertTrue(appears("not in document", "NOT IN DOCUMENT"))


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


class TestExpandSchedule(unittest.TestCase):
    ITEMS = [{"item_id": "old", "prior_strength": 5}, {"item_id": "new", "prior_strength": 5}]

    def test_only_uncovered_items_scheduled(self):
        schedule = expand_schedule([{"item_id": "old", "human": "faithful"}], self.ITEMS, reps=2)
        self.assertEqual({p["item_id"] for p, _, _ in schedule}, {"new"})
        self.assertEqual(len(schedule), 2)

    def test_empty_existing_schedules_everything(self):
        schedule = expand_schedule([], self.ITEMS, reps=2)
        self.assertEqual({p["item_id"] for p, _, _ in schedule}, {"old", "new"})

    def test_fully_covered_schedules_nothing(self):
        existing = [{"item_id": "old"}, {"item_id": "new"}]
        self.assertEqual(expand_schedule(existing, self.ITEMS, reps=2), [])


class TestCustomIds(unittest.TestCase):
    def test_caveat_roundtrip(self):
        cid = encode_caveat_custom_id("grasses", 2, "SOURCE_EXCLUSIVE", 3)
        self.assertEqual(decode_caveat_custom_id(cid),
                          {"fact": "grasses", "severity": 2, "instruction": "SOURCE_EXCLUSIVE", "rep": 3})

    def test_caveat_literal_format(self):
        self.assertEqual(encode_caveat_custom_id("grasses", 2, "SOURCE_EXCLUSIVE", 3),
                          "cv-grasses-s2-SOURCE_EXCLUSIVE-r3")

    def test_abstention_roundtrip(self):
        cid = encode_abstention_custom_id("secondary_dwelling_cap", "WEAK_GROUNDING", 5)
        self.assertEqual(decode_abstention_custom_id(cid),
                          {"item_id": "secondary_dwelling_cap", "instruction": "WEAK_GROUNDING", "rep": 5})

    def test_decode_rejects_wrong_prefix(self):
        with self.assertRaises(ValueError):
            decode_caveat_custom_id(encode_abstention_custom_id("water_boil", "FLAG_INVITING", 0))
        with self.assertRaises(ValueError):
            decode_abstention_custom_id(encode_caveat_custom_id("grasses", 0, "FLAG_INVITING", 0))


class TestCaveatWavePlan(unittest.TestCase):
    INSTR = [("SOURCE_EXCLUSIVE", "sys")]
    LADDER = [{"fact": "grasses", "q": "q", "steps": [{"severity": 0}, {"severity": 1}, {"severity": 2}]}]

    def test_empty_when_all_done(self):
        done = {("m", "SOURCE_EXCLUSIVE", "grasses", s): 5 for s in (0, 1, 2)}
        wave1, wave2 = caveat_wave_plan(done, 5, "m", instructions=self.INSTR, ladders=self.LADDER)
        self.assertEqual((wave1, wave2), ([], []))

    def test_one_rep_per_incomplete_cell(self):
        single = [{"fact": "grasses", "q": "q", "steps": [{"severity": 0}]}]
        wave1, wave2 = caveat_wave_plan({}, 3, "m", instructions=self.INSTR, ladders=single)
        self.assertEqual(wave1, [encode_caveat_custom_id("grasses", 0, "SOURCE_EXCLUSIVE", 0)])
        self.assertEqual(wave2, [encode_caveat_custom_id("grasses", 0, "SOURCE_EXCLUSIVE", 1),
                                  encode_caveat_custom_id("grasses", 0, "SOURCE_EXCLUSIVE", 2)])

    def test_resumes_from_already_count(self):
        single = [{"fact": "grasses", "q": "q", "steps": [{"severity": 0}]}]
        done = {("m", "SOURCE_EXCLUSIVE", "grasses", 0): 2}
        wave1, wave2 = caveat_wave_plan(done, 5, "m", instructions=self.INSTR, ladders=single)
        self.assertEqual(wave1, [encode_caveat_custom_id("grasses", 0, "SOURCE_EXCLUSIVE", 2)])
        self.assertEqual(wave2, [encode_caveat_custom_id("grasses", 0, "SOURCE_EXCLUSIVE", 3),
                                  encode_caveat_custom_id("grasses", 0, "SOURCE_EXCLUSIVE", 4)])

    def test_matches_cell_granularity_on_real_data(self):
        wave1, wave2 = caveat_wave_plan({}, 1, "claude-sonnet-5")
        self.assertEqual(len(wave1), total_cells() // len(MODELS))
        self.assertEqual(wave2, [])


class TestAbstentionWavePlan(unittest.TestCase):
    INSTR = [("SOURCE_EXCLUSIVE", "sys")]
    ITEMS = [{"item_id": "a"}, {"item_id": "b"}]

    def test_one_warm_call_per_instruction_on_real_data(self):
        wave1, wave2 = abstention_wave_plan({}, 1, "claude-sonnet-5")
        self.assertEqual(len(wave1), len(SYSTEM_INSTRUCTIONS))
        self.assertEqual(len(wave2), len(UNANSWERABLE_ITEMS) * len(SYSTEM_INSTRUCTIONS) - len(SYSTEM_INSTRUCTIONS))

    def test_warm_item_is_first_pending_item(self):
        done = {("m", "SOURCE_EXCLUSIVE", "a"): 1}
        wave1, wave2 = abstention_wave_plan(done, 1, "m", instructions=self.INSTR, items=self.ITEMS)
        self.assertEqual(wave1, [encode_abstention_custom_id("b", "SOURCE_EXCLUSIVE", 0)])

    def test_skips_fully_done_instruction(self):
        done = {("m", "SOURCE_EXCLUSIVE", "a"): 1, ("m", "SOURCE_EXCLUSIVE", "b"): 1}
        wave1, wave2 = abstention_wave_plan(done, 1, "m", instructions=self.INSTR, items=self.ITEMS)
        self.assertEqual((wave1, wave2), ([], []))

    def test_second_rep_of_warm_item_goes_to_wave2(self):
        wave1, wave2 = abstention_wave_plan({}, 2, "m", instructions=self.INSTR, items=self.ITEMS)
        self.assertEqual(wave1, [encode_abstention_custom_id("a", "SOURCE_EXCLUSIVE", 0)])
        self.assertIn(encode_abstention_custom_id("a", "SOURCE_EXCLUSIVE", 1), wave2)


class TestManifest(unittest.TestCase):
    def test_structure_and_counts(self):
        m = build_manifest()
        for key in ("generated_at", "git_sha", "run_id", "documents", "instructions", "models",
                    "judge", "n_per_cell", "expected_cells", "perturbation_ladders", "unanswerable_items"):
            self.assertIn(key, m)
        self.assertTrue(m["git_sha"])
        self.assertEqual(set(m["documents"]), set(DOCUMENTS))
        self.assertEqual(len(m["instructions"]), len(SYSTEM_INSTRUCTIONS))
        self.assertEqual(m["expected_cells"]["caveat"], len(MODELS) * len(SYSTEM_INSTRUCTIONS) * total_steps())
        self.assertEqual(m["expected_cells"]["abstention"], len(MODELS) * len(SYSTEM_INSTRUCTIONS) * len(UNANSWERABLE_ITEMS))

    def test_json_roundtrip_stable(self):
        m = build_manifest()
        m.pop("generated_at"), m.pop("run_id"), m.pop("git_sha")
        self.assertEqual(m, json.loads(json.dumps(m)))

    def test_doc_hashes_deterministic(self):
        a, b = build_manifest(), build_manifest()
        self.assertEqual(a["documents"], b["documents"])
        self.assertEqual(a["instructions"], b["instructions"])


class TestChunkedJudgeSink(unittest.TestCase):
    def test_flushes_at_chunk_boundary_in_order(self):
        written = []
        push, flush = _chunked_judge_sink(lambda pair: pair, written.append, chunk=2)
        push("a", 1)
        self.assertEqual(written, [])
        push("b", 2)
        self.assertEqual(written, [("a", 1), ("b", 2)])
        push("c", 3)
        self.assertEqual(written, [("a", 1), ("b", 2)])
        flush()
        self.assertEqual(written, [("a", 1), ("b", 2), ("c", 3)])

    def test_final_flush_on_empty_buffer_is_noop(self):
        written = []
        push, flush = _chunked_judge_sink(lambda pair: pair, written.append, chunk=2)
        flush()
        self.assertEqual(written, [])

    def test_judge_applied_to_each_pair(self):
        written = []
        push, flush = _chunked_judge_sink(lambda pair: pair[1] * 10, written.append, chunk=3)
        for i, cid in enumerate(("a", "b", "c")):
            push(cid, i)
        self.assertEqual(written, [0, 10, 20])


class TestRunAnthropicWave(unittest.TestCase):
    def _succeeded(self, text, stop_reason="end_turn"):
        return mock.Mock(type="succeeded", message=mock.Mock(content=[mock.Mock(type="text", text=text)],
                                                             model="snap-1", stop_reason=stop_reason))

    def _patched(self, results):
        return (mock.patch("harness.submit_anthropic_batch", return_value="b1"),
                mock.patch("harness.poll_anthropic_batch"),
                mock.patch("harness.anthropic_batch_results", return_value=iter(results)))

    def test_succeeded_delivered_before_fallbacks(self):
        events = []
        results = [("a", self._succeeded("A")), ("b", mock.Mock(type="errored"))]
        p1, p2, p3 = self._patched(results)
        with p1, p2, p3:
            _run_anthropic_wave("m", "anthropic", ["a", "b"], "w", lambda m, c: {},
                                lambda cid: (events.append(("sync", cid)), ("B", "snap-2", False))[1],
                                lambda cid, ans, snap, trunc: events.append(("row", cid, ans, snap, trunc)))
        self.assertEqual(events, [("row", "a", "A", "snap-1", False), ("sync", "b"), ("row", "b", "B", "snap-2", False)])

    def test_truncated_flag_threaded_from_stop_reason(self):
        events = []
        results = [("a", self._succeeded("A", stop_reason="max_tokens"))]
        p1, p2, p3 = self._patched(results)
        with p1, p2, p3:
            _run_anthropic_wave("m", "anthropic", ["a"], "w", lambda m, c: {},
                                lambda cid: ("", "", False),
                                lambda cid, ans, snap, trunc: events.append((cid, trunc)))
        self.assertEqual(events, [("a", True)])

    def test_fallback_crash_preserves_succeeded_rows(self):
        written = []
        p1, p2, p3 = self._patched([("a", self._succeeded("A"))])
        def boom(cid):
            raise RuntimeError("credit balance too low")
        with p1, p2, p3:
            with self.assertRaises(RuntimeError):
                _run_anthropic_wave("m", "anthropic", ["a", "b"], "w", lambda m, c: {},
                                    boom, lambda cid, ans, snap, trunc: written.append(cid))
        self.assertEqual(written, ["a"])


class TestPilotSelection(unittest.TestCase):
    def test_filters_to_one_model_and_doc(self):
        models, ladders, items = pilot_selection("claude-sonnet-5", "liquor")
        self.assertEqual(models, [("claude-sonnet-5", "anthropic")])
        self.assertTrue(ladders and all(f["doc"] == "liquor" for f in ladders))
        self.assertTrue(items and all(p["doc"] == "liquor" for p in items))

    def test_does_not_mutate_globals(self):
        n_models, n_ladders, n_items = len(MODELS), len(PERTURBATION_LADDERS), len(UNANSWERABLE_ITEMS)
        pilot_selection("claude-sonnet-5", "liquor")
        self.assertEqual((len(MODELS), len(PERTURBATION_LADDERS), len(UNANSWERABLE_ITEMS)),
                         (n_models, n_ladders, n_items))

    def test_unknown_model_exits(self):
        with self.assertRaises(SystemExit):
            pilot_selection("gpt-9", "liquor")

    def test_unknown_doc_exits(self):
        with self.assertRaises(SystemExit):
            pilot_selection("claude-sonnet-5", "zoning")


class TestUnitCounts(unittest.TestCase):
    ROWS = [{"fact": "a", "stance": "questioned"}, {"fact": "a", "stance": "silent"},
            {"fact": "b", "stance": "questioned"}]

    def test_counts_per_unit(self):
        per = unit_counts(self.ROWS, lambda r: r["stance"] == "questioned")
        self.assertEqual(per, {"a": (1, 2), "b": (1, 1)})

    def test_rate_map_fills_missing_units_with_none(self):
        rates = unit_rate_map(self.ROWS, lambda r: r["stance"] == "questioned", ["a", "b", "c"])
        self.assertEqual(rates, {"a": 0.5, "b": 1.0, "c": None})


class TestBuildBatchMessageParams(unittest.TestCase):
    def test_cache_control_only_on_passage(self):
        params = build_batch_message_params("claude-sonnet-5", "sys", "q?", "DOC")
        content = params["messages"][0]["content"]
        self.assertEqual(content[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertNotIn("cache_control", content[1])

    def test_blocks_concatenate_to_sync_prompt(self):
        params = build_batch_message_params("claude-sonnet-5", "sys", "q?", "DOC")
        content = params["messages"][0]["content"]
        self.assertEqual(content[0]["text"] + content[1]["text"], "Passage:\nDOC\n\nQuestion: q?")


class TestSweepSpecs(unittest.TestCase):
    def test_codec_roundtrip_and_done_key_arity(self):
        for spec, unit in ((CAVEAT_SWEEP, ("factx", 3)), (ABSTENTION_SWEEP, ("itemx",)), (ABSENCE_SWEEP, ("factx",))):
            cid = spec.encode(unit, "FLAG_INVITING", 2)
            self.assertEqual(spec.decode(cid), (unit, "FLAG_INVITING", 2))
            self.assertEqual(len(spec.done_fields), 2 + len(unit))

    def test_units_match_datasets(self):
        self.assertEqual(len(CAVEAT_SWEEP.units(CAVEAT_SWEEP.dataset())), total_steps())
        self.assertEqual(len(ABSTENTION_SWEEP.units(ABSTENTION_SWEEP.dataset())), len(UNANSWERABLE_ITEMS))
        self.assertEqual(len(ABSENCE_SWEEP.units(ABSENCE_SWEEP.dataset())), len(PERTURBATION_LADDERS))

    def test_prompt_doc_matches_judged_doc(self):
        for spec, unit in ((ABSENCE_SWEEP, (PERTURBATION_LADDERS[0]["fact"],)),
                           (ABSTENTION_SWEEP, (UNANSWERABLE_ITEMS[0]["item_id"],))):
            with mock.patch("harness.abstention_judge", return_value=(True, "r", "s")) as j:
                q, doc = spec.prompt(unit)
                spec.row("m", "openai", "FLAG_INVITING", unit, doc, "answer", "snap", 0)
                self.assertEqual(j.call_args[0][0], q)
                self.assertEqual(j.call_args[0][1], doc)


class TestRunSweep(unittest.TestCase):
    def _fake_spec(self, path):
        units = [("u1",), ("u2",)]
        return SweepSpec(
            "fake", path, ["model", "instruction", "u"],
            lambda n: True, lambda: units, lambda ds: ds, "cell",
            lambda u, i, r: f"fk-{u[0]}-{i}-r{r}",
            lambda cid: ((cid.split("-")[1],), cid.split("-")[2], int(cid.split("-")[3][1:])),
            lambda u: ("q?", "DOC"),
            lambda model, prov, iname, u, doc, answer, snapshot, rep, truncated=False:
                {"model": model, "instruction": iname, "u": u[0], "rep": rep, "answer": answer,
                 "truncated": truncated, "label": "ok"},
            lambda u: u[0], lambda u: u[0], lambda: None)

    def test_generic_wave_plan_both_strategies(self):
        spec = self._fake_spec("unused")
        done = {("m", "I", "u1"): 1}
        w1, w2 = _sweep_wave_plan(spec, done, 2, "m", [("I", "sys")], [("u1",), ("u2",)])
        self.assertEqual(w1, ["fk-u1-I-r1", "fk-u2-I-r0"])
        self.assertEqual(w2, ["fk-u2-I-r1"])
        w1, w2 = _sweep_wave_plan(spec._replace(warm="instruction"), done, 2, "m", [("I", "sys")], [("u1",), ("u2",)])
        self.assertEqual(w1, ["fk-u1-I-r1"])
        self.assertEqual(w2, ["fk-u2-I-r0", "fk-u2-I-r1"])

    def _ok_rec(self, cid):
        return (cid, {"custom_id": cid, "error": None,
                      "response": {"status_code": 200, "body": {"model": "snap", "output": [
                          {"type": "message", "content": [{"type": "output_text", "text": "ans"}]}]}}})

    def test_run_sweep_writes_then_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.jsonl")
            spec = self._fake_spec(path)
            submitted = []
            def fake_submit(reqs, endpoint="/v1/responses"):
                submitted.append([cid for cid, _ in reqs])
                return "b1"
            def fake_results(bid):
                for cid in submitted[-1]:
                    yield self._ok_rec(cid)
            with mock.patch("harness.MODELS", [("m", "openai")]), \
                 mock.patch("harness.SYSTEM_INSTRUCTIONS", [("I", "sys")]), \
                 mock.patch("harness.INSTR_BY_NAME", {"I": "sys"}), \
                 mock.patch("harness.submit_openai_batch", side_effect=fake_submit), \
                 mock.patch("harness.poll_openai_batch", return_value=types.SimpleNamespace(status="completed")), \
                 mock.patch("harness.openai_batch_results", side_effect=fake_results):
                _run_sweep(spec, 1)
                self.assertEqual(len(submitted), 1)
                with open(path) as f:
                    rows = [json.loads(l) for l in f]
                self.assertEqual([r["u"] for r in rows], ["u1", "u2"])
                self.assertEqual([r["answer"] for r in rows], ["ans", "ans"])
                _run_sweep(spec, 1)
                self.assertEqual(len(submitted), 1)
                with open(path) as f:
                    self.assertEqual(len(f.readlines()), 2)

    def test_run_sweep_openai_error_falls_back_to_sync(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.jsonl")
            spec = self._fake_spec(path)
            submitted = []
            def fake_submit(reqs, endpoint="/v1/responses"):
                submitted.append([cid for cid, _ in reqs])
                return "b1"
            def fake_results(bid):
                cids = submitted[-1]
                yield self._ok_rec(cids[0])
                yield (cids[1], {"custom_id": cids[1], "response": None, "error": {"message": "boom"}})
            with mock.patch("harness.MODELS", [("m", "openai")]), \
                 mock.patch("harness.SYSTEM_INSTRUCTIONS", [("I", "sys")]), \
                 mock.patch("harness.INSTR_BY_NAME", {"I": "sys"}), \
                 mock.patch("harness.submit_openai_batch", side_effect=fake_submit), \
                 mock.patch("harness.poll_openai_batch", return_value=types.SimpleNamespace(status="completed")), \
                 mock.patch("harness.openai_batch_results", side_effect=fake_results), \
                 mock.patch("harness.with_retry", return_value=("synced", "snap2", False)) as wr:
                _run_sweep(spec, 1)
                self.assertEqual(wr.call_count, 1)
                with open(path) as f:
                    answers = sorted(json.loads(l)["answer"] for l in f)
                self.assertEqual(answers, ["ans", "synced"])

    def test_run_sweep_stamps_truncated_from_body_status(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.jsonl")
            spec = self._fake_spec(path)
            submitted = []
            def fake_submit(reqs, endpoint="/v1/responses"):
                submitted.append([cid for cid, _ in reqs])
                return "b1"
            def fake_results(bid):
                cids = submitted[-1]
                cid, rec = self._ok_rec(cids[0])
                rec["response"]["body"]["status"] = "incomplete"
                yield cid, rec
                yield self._ok_rec(cids[1])
            with mock.patch("harness.MODELS", [("m", "openai")]), \
                 mock.patch("harness.SYSTEM_INSTRUCTIONS", [("I", "sys")]), \
                 mock.patch("harness.INSTR_BY_NAME", {"I": "sys"}), \
                 mock.patch("harness.submit_openai_batch", side_effect=fake_submit), \
                 mock.patch("harness.poll_openai_batch", return_value=types.SimpleNamespace(status="completed")), \
                 mock.patch("harness.openai_batch_results", side_effect=fake_results):
                _run_sweep(spec, 1)
                with open(path) as f:
                    rows = [json.loads(l) for l in f]
                self.assertEqual({r["u"]: r["truncated"] for r in rows}, {"u1": True, "u2": False})

    def test_run_sweep_aborts_on_wholesale_batch_failure(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.jsonl")
            spec = self._fake_spec(path)
            failed = types.SimpleNamespace(status="failed", errors=types.SimpleNamespace(
                data=[types.SimpleNamespace(message="Enqueued token limit reached")]))
            with mock.patch("harness.MODELS", [("m", "openai")]), \
                 mock.patch("harness.SYSTEM_INSTRUCTIONS", [("I", "sys")]), \
                 mock.patch("harness.INSTR_BY_NAME", {"I": "sys"}), \
                 mock.patch("harness.submit_openai_batch", return_value="b1"), \
                 mock.patch("harness.poll_openai_batch", return_value=failed), \
                 mock.patch("harness.with_retry") as wr:
                with self.assertRaises(SystemExit) as ctx:
                    _run_sweep(spec, 1)
                self.assertIn("Enqueued token limit reached", str(ctx.exception))
                wr.assert_not_called()
                self.assertFalse(os.path.exists(path) and open(path).read())


class TestOpenAIBatch(unittest.TestCase):
    def test_body_terra_no_reasoning_no_format(self):
        b = config.build_openai_batch_body("gpt-5.6-terra", "sys", "in")
        self.assertEqual(b["model"], "gpt-5.6-terra")
        self.assertEqual(b["instructions"], "sys")
        self.assertEqual(b["input"], "in")
        self.assertEqual(b["max_output_tokens"], 2000)
        self.assertNotIn("reasoning", b)
        self.assertNotIn("text", b)

    def test_body_judge_reasoning_and_format(self):
        fmt = {"type": "json_schema", "name": "verdict"}
        b = config.build_openai_batch_body("gpt-5.4-mini", "s", "i", max_output_tokens=2048, text_format=fmt)
        self.assertEqual(b["reasoning"], {"effort": "low"})
        self.assertEqual(b["text"], {"format": fmt})
        self.assertEqual(b["max_output_tokens"], 2048)

    @mock.patch("config.openai_client")
    def test_submit_writes_jsonl_and_creates_batch(self, oc):
        client = oc.return_value
        client.files.create.return_value = types.SimpleNamespace(id="file-1")
        client.batches.create.return_value = types.SimpleNamespace(id="batch-1")
        body = config.build_openai_batch_body("gpt-5.6-terra", "sys", "in")
        bid = config.submit_openai_batch([("cid-1", body)])
        self.assertEqual(bid, "batch-1")
        _, fkwargs = client.files.create.call_args
        _, data = fkwargs["file"]
        line = json.loads(data.decode())
        self.assertEqual(line["custom_id"], "cid-1")
        self.assertEqual(line["method"], "POST")
        self.assertEqual(line["url"], "/v1/responses")
        self.assertEqual(line["body"]["model"], "gpt-5.6-terra")
        self.assertEqual(fkwargs["purpose"], "batch")
        _, bkwargs = client.batches.create.call_args
        self.assertEqual(bkwargs["input_file_id"], "file-1")
        self.assertEqual(bkwargs["endpoint"], "/v1/responses")
        self.assertEqual(bkwargs["completion_window"], "24h")

    @mock.patch("config.time.sleep")
    @mock.patch("config.openai_client")
    def test_poll_loops_until_terminal(self, oc, sleep):
        client = oc.return_value
        client.batches.retrieve.side_effect = [types.SimpleNamespace(status="in_progress"),
                                               types.SimpleNamespace(status="completed")]
        seen = []
        b = config.poll_openai_batch("b1", poll_interval=0, on_poll=lambda x: seen.append(x.status))
        self.assertEqual(b.status, "completed")
        self.assertEqual(seen, ["in_progress", "completed"])
        sleep.assert_called_once()

    @mock.patch("config.openai_client")
    def test_results_yields_success_and_error_lines(self, oc):
        client = oc.return_value
        client.batches.retrieve.return_value = types.SimpleNamespace(output_file_id="out", error_file_id="err")
        texts = {"out": '{"custom_id":"a","response":{"status_code":200,"body":{"x":1}},"error":null}\n',
                 "err": '{"custom_id":"b","response":null,"error":{"message":"boom"}}\n'}
        client.files.content.side_effect = lambda fid: types.SimpleNamespace(text=texts[fid])
        got = dict(config.openai_batch_results("b1"))
        self.assertEqual(got["a"]["response"]["status_code"], 200)
        self.assertIsNone(got["a"]["error"])
        self.assertEqual(got["b"]["error"]["message"], "boom")

    def test_extract_text_skips_reasoning_item(self):
        body = {"model": "m", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello "},
                {"type": "output_text", "text": "world"}]}]}
        self.assertEqual(config.extract_openai_text(body), "hello world")
        self.assertEqual(config.extract_openai_text({"output": []}), "")

    def test_chunks_split_by_token_budget(self):
        reqs = [(f"c{i}", {"instructions": "x" * 400, "input": "y" * 400}) for i in range(5)]
        with mock.patch("harness.OPENAI_BATCH_TOKEN_BUDGET", 500), \
             mock.patch("harness.OPENAI_BATCH_MAX_REQUESTS", 100):
            chunks = list(_openai_batch_chunks(reqs))
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])

    def test_chunks_count_max_output_tokens_toward_budget(self):
        reqs = [(f"c{i}", {"instructions": "", "input": "", "max_output_tokens": 200}) for i in range(5)]
        with mock.patch("harness.OPENAI_BATCH_TOKEN_BUDGET", 500), \
             mock.patch("harness.OPENAI_BATCH_MAX_REQUESTS", 100):
            chunks = list(_openai_batch_chunks(reqs))
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])

    def test_chunks_split_by_max_requests(self):
        reqs = [(f"c{i}", {"instructions": "", "input": ""}) for i in range(5)]
        with mock.patch("harness.OPENAI_BATCH_TOKEN_BUDGET", 10 ** 9), \
             mock.patch("harness.OPENAI_BATCH_MAX_REQUESTS", 2):
            chunks = list(_openai_batch_chunks(reqs))
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
