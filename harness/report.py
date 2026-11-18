"""Turn metrics.json into a readable report.md."""
from __future__ import annotations

from pathlib import Path

FAMILY_ORDER = ["teleport", "teleport_jitter", "const_speed", "const_time", "ease_tween", "bezier", "wind",
                "fitts_aware", "replay", "sim_bot"]
TIER = {"teleport": 0, "teleport_jitter": 0, "const_speed": 1, "const_time": 1, "ease_tween": 1,
        "bezier": 2, "wind": 2, "fitts_aware": 3, "replay": 3, "sim_bot": 3}


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def write_report(M: dict, path: Path) -> None:
    L: list[str] = []
    L += [f"# Harness report: {M['name']}", "",
          f"Real people: **{M['n_people']}** ({M['n_trials']} trials, Findlater & Zhang 2020). "
          "All numbers are on held-out people (5 folds split by person). "
          "Bots click through the same targets as each human window.", ""]

    L += ["## Gates", "", "| gate | what | value | need | result |", "|---|---|---|---|---|"]
    for g in M["gates"]:
        res = "PASS" if g["passed"] else ("FAIL" if g["blocking"] else
                                         ("OPEN (target not met)" if g.get("open") else "miss (stretch)"))
        L.append(f"| {g['id']} | {g['desc']} | {g['value']:.3f} | {g['op']} {g['limit']} | {res} |")
    L.append("")

    L += ["## Detection rate by bot family (alarm threshold set for 1% false positives on training people)", ""]
    for k, dets in M["detection"].items():
        names = list(dets)
        L += [f"### {k[1:]} clicks per decision", "",
              "| bot family | tier | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        fams = [f for f in FAMILY_ORDER if f in dets[names[0]]["families"]]
        for fam in fams:
            L.append(f"| {fam} | {TIER.get(fam, '?')} | " +
                     " | ".join(_pct(dets[n]["families"][fam]["tpr"]) for n in names) + " |")
        L.append("| **false positives (real people)** | | " + " | ".join(f"**{_pct(dets[n]['fpr'])}**" for n in names) + " |")
        L.append("")

    if "detection_age" in M:
        L += ["## Does conditioning the Fitts tests on age help?", "",
              "Time-scale statistics are divided by the expected slowing at the account's age before testing. "
              "A bot inherits the age of the account it runs on.", "",
              "| clicks | variant | FPR all | worst age-band FPR | tier0 | tier1 | tier2 | tier3 |", "|---|---|---|---|---|---|---|---|"]
        for k, v in M["detection_age"].items():
            for name, r in v.items():
                t = r["tiers"]
                L.append(f"| {k[1:]} | {name} | {_pct(r['fpr'])} | {_pct(r['worst_band_fpr'])} | " +
                         " | ".join(_pct(t.get(f'tier{i}', float('nan'))) for i in range(4)) + " |")
        L.append("")

    L += ["## Who gets falsely flagged (25 clicks)", ""]
    for name, sg in M["subgroups"].items():
        L += [f"### {name}", "", "| group | windows | false-positive rate |", "|---|---|---|"]
        for g, v in sg.items():
            L.append(f"| {g} | {v['n']} | {_pct(v['fpr'])} |")
        L.append("")

    L += ["## Supervised reference (gradient boosting, 25 clicks)", "",
          "Trained on humans plus bots. *Leave-family-out*: the family in the row was removed from training, "
          "so the number is performance on an attack never seen before.", ""]
    for gname in ("fitts", "fitts_kin"):
        if gname not in M["supervised"]:
            continue
        sup, lofo = M["supervised"][gname], M["supervised"][gname + "_leave_family_out"]
        L += [f"### features: {gname}  (FPR {_pct(sup['fpr'])})", "",
              "| bot family | tier | seen in training | never seen (leave-family-out) | FPR in that run |", "|---|---|---|---|---|"]
        for fam in [f for f in FAMILY_ORDER if f in sup["families"]]:
            L.append(f"| {fam} | {TIER.get(fam, '?')} | {_pct(sup['families'][fam]['tpr'])} | "
                     f"{_pct(lofo[fam]['tpr'])} | {_pct(lofo[fam]['fpr'])} |")
        L.append("")

    A = M["age"]
    f = A["fit"]
    L += ["## Age model", "",
          f"Fitted on {A['n_people']} people aged 18-83; cross-validated by person. Chosen variant: **{A['chosen']}**"
          + ("" if A.get("identifiable") else " (the data cannot separate ageing of the intercept from ageing of the "
             "slope: the task only spans 1.9-4.1 bits, so proportional slowing is assumed)") + ".", "",
          "| variant | person-level median error | age-band mean error | worst age band |", "|---|---|---|---|"]
    for v, s in A["variants"].items():
        L.append(f"| {v} | {s['person_median_abs_pct']:.1f}% | {s['band_mean_abs_pct']:.1f}% | {s['band_max_abs_pct']:.1f}% |")
    L += ["", "Held-out age-band means (chosen variant):", "",
          "| age band | people | observed MT | predicted MT | error |", "|---|---|---|---|---|"]
    for b, s in A["chosen_scores"]["bands"].items():
        L.append(f"| {b} | {s['n']} | {s['mt_obs_ms']:.0f} ms | {s['mt_pred_ms']:.0f} ms | {s['err_pct']:+.1f}% |")
    L += ["",
          f"- reference (age <= 30, mouse, no motor impairment): MT = {f['a_ref_ms']:.0f} ms + {f['b_ref_ms_per_bit']:.0f} ms/bit x ID",
          f"- ageing after 30: intercept {f['ageing_a_pct_per_decade']:+.1f}% and slope {f['ageing_b_pct_per_decade']:+.1f}% per decade",
          f"- touchpad x{f['touchpad_mult']:.2f}, self-reported motor impairment x{f['motor_impairment_mult']:.2f}",
          f"- button hold time: {f['hold_ref_ms']:.0f} ms at <= 30, {f['hold_ageing_pct_per_decade']:+.1f}% per decade",
          f"- spread between people of the same age: s.d. of log MT = {f['between_person_sd_log']:.2f}", "",
          "Predicted values by age (D = 400 px, W = 32 px, ID = 3.75 bits):", ""]
    has_cps = any("max_cps" in r for r in A["table"])
    L += ["| age | a (ms) | b (ms/bit) | MT (ms) | 95% of people | throughput (bits/s) | hold (ms) |" + (" max clicks/s |" if has_cps else ""),
          "|---|---|---|---|---|---|---|" + ("---|" if has_cps else "")]
    for r in A["table"]:
        L.append(f"| {r['age']} | {r['a_ms']:.0f} | {r['b_ms_per_bit']:.0f} | {r['mt_ms_D400_W32']:.0f} | "
                 f"{r['mt_lo_ms']:.0f}-{r['mt_hi_ms']:.0f} | {r['throughput']:.2f} | {r['hold_ms']:.0f} |"
                 + (f" {r['max_cps']:.1f} |" if has_cps and "max_cps" in r else (" |" if has_cps else "")))
    if "literature" in A:
        L += ["", "Literature-based parts (within-study ratios, leave-one-study-out):", ""]
        for m, v in A["literature"].items():
            if "error" in v:
                L.append(f"- {m}: not fitted ({v['error']})")
            else:
                fac = ", ".join(f"age {a}: x{x:.2f}" for a, x in v["factor_at"].items())
                L.append(f"- {m}: {v['n_studies']} studies / {v['n_groups']} groups, ages {v['age_min']:.0f}-{v['age_max']:.0f}, "
                         f"leave-one-study-out error {100 * v['loso_rmse_log']:.0f}% (log); "
                         f"ageing {100 * v['ageing_per_year']:+.2f}%/year after 30; factor vs age 25: {fac}")
    L.append("")
    path.write_text("\n".join(L))
