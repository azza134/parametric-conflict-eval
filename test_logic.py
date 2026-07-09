# Tests to verify the functions are working as intended. No API key required.
import json
import math
import os
import tempfile
import unittest
from unittest import mock

from config import (perturb, with_retry, DOCUMENTS, doc_text, openai_reasoning, MODELS, FLAG_INVITING, SOURCE_EXCLUSIVE,
                    SOURCE_EXCLUSIVE_FLAG_INVITING, SYSTEM_INSTRUCTIONS,
                    appears, passage, step_doc, build_batch_message_params)
from harness import (wilson_interval, PERTURBATION_LADDERS, SEVERITIES, validate_ladders,
                     total_steps, total_cells, classify, lexical_caveat, UNANSWERABLE_ITEMS, validate_items,
                     load_done, tradeoff_rows, PRIOR_STRENGTHS, cluster_icc, vector_cells,
                     probe_targets, _probe_row, probe_item_rates, measured_prior_bins, prior_bin, prior_bin_label,
                     encode_caveat_custom_id, decode_caveat_custom_id,
                     encode_abstention_custom_id, decode_abstention_custom_id,
                     caveat_wave_plan, abstention_wave_plan, concurrent_map)
from judge import (cohens_kappa, FAITHFUL, UNGROUNDED,
                   judge_gate, anchor_disagreements, GATE_PASS, GATE_FAIL, KAPPA_THRESHOLD,
                   QUESTIONED, SILENT, ENDORSED, DECLINED, CAVEAT_LABELS, CAVEAT_SCHEMA, build_caveat_prompt,
                   CORROBORATION_LABELS,
                   gold_schedule, _meta_evaluate)


class TestPerturb(unittest.TestCase):
    def test_replaces(self):
        self.assertEqual(perturb("every 20 persons", [("every 20", "every 13")]), "every 13 persons") # when 20 is replaced with 13 does it equal argument 3

    def test_raises_on_noop(self):
        with self.assertRaises(AssertionError):
            perturb("nothing to change here", [("absent token", "x")]) # argument 1 is a phrase that doesn't exist in passage, thus nothing to replace, raising the assertion error for the perturb function

    def test_raises_when_one_of_several_replacements_misses(self):
        with self.assertRaises(AssertionError):
            perturb("every 20 persons", [("every 20", "every 25"), ("part of 20 persons", "part of 25 persons")])


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


class TestInstructions(unittest.TestCase):
    def test_instruction_names_and_order(self):
        self.assertEqual([name for name, _ in SYSTEM_INSTRUCTIONS],
                         ["SOURCE_EXCLUSIVE", "FLAG_INVITING", "WEAK_GROUNDING", "SOURCE_EXCLUSIVE_FLAG_INVITING"])

    def test_four_distinct_instructions(self):
        self.assertEqual(len({t for _, t in SYSTEM_INSTRUCTIONS}), 4)

    def test_permissive_invites_flagging(self):
        self.assertIn("flag", FLAG_INVITING.lower())

    def test_composed_arm_reuses_source_exclusive_verbatim(self):
        self.assertTrue(SOURCE_EXCLUSIVE_FLAG_INVITING.startswith(SOURCE_EXCLUSIVE))
        self.assertIn("flag your concern", SOURCE_EXCLUSIVE_FLAG_INVITING)

    def test_no_hyphens_in_instruction_names(self):
        for name, _ in SYSTEM_INSTRUCTIONS:
            self.assertNotIn("-", name)

    def test_no_hyphens_in_fact_names(self):
        for fact in PERTURBATION_LADDERS:
            self.assertNotIn("-", fact["fact"])


class TestClusterIcc(unittest.TestCase):
    def test_all_or_nothing_clusters(self):
        p, rho, neff = cluster_icc([(0, 8), (0, 8), (0, 8), (8, 8), (8, 8), (8, 8)])
        self.assertEqual(p, 0.5)
        self.assertEqual(rho, 1.0)
        self.assertAlmostEqual(neff, 6.0)

    def test_degenerate_cell_has_no_icc(self):
        self.assertEqual(cluster_icc([(0, 8)] * 6), (0.0, None, None))
        self.assertEqual(cluster_icc([(8, 8)] * 6), (1.0, None, None))

    def test_uncorrelated_reps_keep_full_n(self):
        p, rho, neff = cluster_icc([(1, 8), (0, 8), (0, 8), (0, 8), (1, 8), (0, 8)])
        self.assertEqual(rho, 0.0)
        self.assertAlmostEqual(neff, 48.0)


class TestMeasuredPriorBins(unittest.TestCase):
    def test_no_probe_file_yields_no_bins(self):
        with mock.patch("harness.PROBE_RESULTS", "no_such_probe_file.jsonl"):
            self.assertEqual(probe_item_rates(), {})
            self.assertEqual(measured_prior_bins(), {})

    def test_partial_probe_coverage_yields_no_bins(self):
        with mock.patch("harness.probe_item_rates", return_value={"water_boil": 1.0}):
            self.assertEqual(measured_prior_bins(), {})

    def test_fixed_edges_do_not_move_with_the_sample(self):
        self.assertEqual(prior_bin(0.0), 0)
        self.assertEqual(prior_bin(0.24), 0)
        self.assertEqual(prior_bin(0.25), 1)
        self.assertEqual(prior_bin(0.5), 2)
        self.assertEqual(prior_bin(0.75), 3)
        self.assertEqual(prior_bin(1.0), 3)
        self.assertEqual(prior_bin_label(0), "0.00-0.25")
        self.assertEqual(prior_bin_label(3), "0.75-1.00")

    def test_lopsided_sample_yields_lopsided_bins(self):
        rates = {p["item_id"]: 0.9 for p in UNANSWERABLE_ITEMS}
        with mock.patch("harness.probe_item_rates", return_value=rates):
            bins = measured_prior_bins()
        self.assertEqual(len(bins), len(UNANSWERABLE_ITEMS))
        self.assertEqual({b for b, _ in bins.values()}, {3})

    def test_even_spread_fills_all_bins(self):
        n = len(UNANSWERABLE_ITEMS)
        rates = {p["item_id"]: i / (n - 1) for i, p in enumerate(UNANSWERABLE_ITEMS)}
        with mock.patch("harness.probe_item_rates", return_value=rates):
            bins = measured_prior_bins()
        counts = {}
        for b, label in bins.values():
            counts[b] = counts.get(b, 0) + 1
        self.assertEqual(set(counts), {0, 1, 2, 3})
        self.assertEqual(sum(counts.values()), n)

    def test_retired_items_in_probe_file_are_ignored(self):
        rates = {p["item_id"]: 0.9 for p in UNANSWERABLE_ITEMS}
        rates["some_retired_item"] = 0.1
        with mock.patch("harness.probe_item_rates", return_value=rates):
            bins = measured_prior_bins()
        self.assertEqual(set(bins), {p["item_id"] for p in UNANSWERABLE_ITEMS})

    def test_rates_pool_reps_and_models(self):
        rows = [{"kind": "item", "name": "x", "reports_expected": True},
                {"kind": "item", "name": "x", "reports_expected": False},
                {"kind": "fact", "name": "y", "reports_expected": True}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with mock.patch("harness.PROBE_RESULTS", f.name):
            self.assertEqual(probe_item_rates(), {"x": 0.5})


class TestEffortConvention(unittest.TestCase):
    def test_v1_gpt54_candidates_stay_pinned_low(self):
        self.assertEqual(openai_reasoning("gpt-5.4-nano"), {"reasoning": {"effort": "low"}})
        self.assertEqual(openai_reasoning("gpt-5.4-mini"), {"reasoning": {"effort": "low"}})

    def test_new_models_run_at_vendor_default(self):
        self.assertEqual(openai_reasoning("gpt-5.6-terra"), {})
        self.assertEqual(openai_reasoning("gpt-4o-mini"), {})


class TestPriorProbe(unittest.TestCase):
    def test_forty_targets_covering_all_facts_and_items(self):
        targets = probe_targets()
        self.assertEqual(len(targets), len(PERTURBATION_LADDERS) + len(UNANSWERABLE_ITEMS))
        self.assertEqual({t["kind"] for t in targets}, {"fact", "item"})
        for t in targets:
            self.assertIn(t["doc"], DOCUMENTS)
            self.assertTrue(t["q"] and t["expected"])

    def test_item_targets_use_prior_strength_as_rating(self):
        t = next(t for t in probe_targets() if t["name"] == "water_boil")
        self.assertEqual(t["prior_rating"], 5)

    def test_probe_row_lexical_flags(self):
        t = {"kind": "fact", "name": "x", "doc": "consent", "q": "?", "expected": "20", "prior_rating": 3}
        row = _probe_row("m", "openai", t, "The value is 20 persons.")
        self.assertTrue(row["reports_expected"])
        self.assertFalse(row["says_dont_know"])
        row = _probe_row("m", "openai", t, "I do not know.")
        self.assertFalse(row["reports_expected"])
        self.assertTrue(row["says_dont_know"])


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


class TestLadders(unittest.TestCase):
    def test_all_perturbations_apply_to_document(self):
        self.assertEqual(validate_ladders(), [])

    def test_every_fact_ladders_0_to_5(self):
        for fact in PERTURBATION_LADDERS:
            self.assertEqual([s["severity"] for s in fact["steps"]], SEVERITIES)

    def test_one_hundred_twenty_steps_total(self):
        self.assertEqual(total_steps(), 120)

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
            self.assertFalse(appears(p["parametric_answer"], doc_text(p["doc"])), p["item_id"])

    def test_ten_items_after_top_bin_cut(self):
        self.assertEqual(len(UNANSWERABLE_ITEMS), 10)

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


if __name__ == "__main__":
    unittest.main()
