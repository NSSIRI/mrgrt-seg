"""Builds paper/Article_Ablation_6OAR_v1.md : text template + tables generated from the verified CSVs.

    .venv\\Scripts\\python.exe paper/build_article_ablation.py

Sources (read-only):
  results/analyse_complete/tableaux/02_*.csv, 03_*.csv         phase 2 (E1, E2, E3, 187 patients)
  phase3_ablation/analyse/stats_*.csv, ablation_masks.csv      ablation (A1-A3, seeds)
  results/phase3_ablation/*/abl18m-*/ablation_eval/            per-run metrics (HDmax)
  phase4_6oar/analyse/p4_*.csv                                 6 OAR (D115)
  phase4_6oar/thorax/stats_thorax.csv, couverture_thorax.csv   6 OAR restricted to the lung slices
  phase3_ablation/xai_matched/analyse/xai*_stats_apparies.csv  matched-layer Seg-Grad-CAM (D187, D115)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AC = ROOT / "results/analyse_complete/tableaux"
AB = ROOT / "phase3_ablation/analyse"
P4 = ROOT / "phase4_6oar/analyse"
ORG = ["poumon_g", "poumon_d", "coeur", "oesophage"]
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}


def f3(x):
    return f"{x:.3f}"


def ci(r, k=3):
    return f"{r.med_diff:+.{k}f} [{r.ic_bas:.{k}f}, {r.ic_haut:.{k}f}]"


def p_fmt(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def table_phase2():
    d = pd.read_csv(AC / "02_descriptifs_E1_E2_E3_187.csv")
    lines = ["| OAR | E1 U-Net 7.9 M | E3 U-Net 18.8 M | E2 SegResNet 18.8 M |", "|---|---|---|---|"]
    for metric, lab, k in [("dsc", "DSC", 3), ("hd95_mm", "HD95 (mm)", 1), ("sdsc_2mm", "Surface DSC 2 mm", 3)]:
        for o in ORG:
            cells = []
            for m in ["E1_UNet_base", "E3_UNet_opt", "E2_SegResNet"]:
                r = d[(d.modele == m) & (d.organe == o) & (d.metrique == metric)].iloc[0]
                cells.append(f"{r.mediane:.{k}f} [{r.q1:.{k}f}–{r.q3:.{k}f}]")
            lines.append(f"| {lab} — {EN[o]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_paired_phase2():
    p = pd.read_csv(AC / "03_comparaisons_appariees_187.csv")
    lines = ["| OAR | E2 − E3 (DSC) | E3 − E1 (DSC) | E2 − E3 (HD95, mm) | E3 − E1 (HD95, mm) |", "|---|---|---|---|---|"]
    for o in ORG:
        c = []
        for metric, k in [("dsc", 3), ("hd95_mm", 1)]:
            for comp in ["E2_SegResNet - E3_UNet_opt", "E3_UNet_opt - E1_UNet_base"]:
                r = p[(p.metrique == metric) & (p.organe == o) & (p.comparaison == comp)].iloc[0]
                c.append((metric, comp, f"{r.mediane_diff:+.{k}f} [{r.ic95_bas:.{k}f}, {r.ic95_haut:.{k}f}]"))
        lines.append(f"| {EN[o]} | {c[0][2]} | {c[1][2]} | {c[2][2]} | {c[3][2]} |")
    return "\n".join(lines)


def table_seeds():
    q = pd.read_csv(AB / "stats_q1_graines.csv")
    q = q[q.metrique == "dsc"]
    labs = ["E2-E3 graine 42", "E2-E3 graine 123", "effet graine E3 (s123-s42)", "effet graine E2 (s123-s42)"]
    lines = ["| OAR | E2 − E3, seed 42 | E2 − E3, seed 123 | Seed effect E3 (123 − 42) | Seed effect E2 (123 − 42) |",
             "|---|---|---|---|---|"]
    for o in ORG:
        cells = [ci(q[(q.organe == o) & (q.comparaison == lab)].iloc[0]) for lab in labs]
        lines.append(f"| {EN[o]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_ablation():
    q = pd.read_csv(AB / "stats_q2_variantes.csv")
    g = pd.read_csv(AB / "stats_part_ecart.csv")
    names = {"A1": "A1 U-Net + residual units", "A2": "A2 U-Net + GroupNorm", "A3": "A3 SegResNet + InstanceNorm"}
    lines = ["| Variant | OAR | Δ vs E3 (DSC) | Δ vs E2 (DSC) | Gap closed (DSC) | Δ vs E3 (HD95, mm) |",
             "|---|---|---|---|---|---|"]
    for v in ["A1", "A2", "A3"]:
        for o in ORG:
            e3 = q[(q.metrique == "dsc") & (q.variante == v) & (q.reference == "E3") & (q.organe == o)].iloc[0]
            e2 = q[(q.metrique == "dsc") & (q.variante == v) & (q.reference == "E2") & (q.organe == o)].iloc[0]
            h3 = q[(q.metrique == "hd95_mm") & (q.variante == v) & (q.reference == "E3") & (q.organe == o)].iloc[0]
            gp = g[(g.metrique == "dsc") & (g.variante == v) & (g.organe == o)].iloc[0]
            lines.append(f"| {names[v] if o == ORG[0] else ''} | {EN[o]} | {ci(e3)} | {ci(e2)} | "
                         f"{gp.part_comblee:.2f} [{gp.ic_bas:.2f}, {gp.ic_haut:.2f}] | {ci(h3, 1)} |")
    return "\n".join(lines)


def hdmax():
    c = pd.read_csv(ROOT / "results/phase2_eval/analyse/combined_long.csv")
    c = c[(c.space == "native") & c.model.isin(["E3_UNet_opt", "E2_SegResNet"])]
    c["m"] = c.model.map({"E3_UNet_opt": "E3_s42", "E2_SegResNet": "E2_s42"})
    names = {("unet_res", 42): "A1", ("unet_gn", 42): "A2", ("segresnet_in", 42): "A3"}
    runs = []
    for f in (ROOT / "results/phase3_ablation").glob("*/abl18m-*/ablation_eval/metrics_long.csv"):
        if "ctrl" in str(f):
            continue
        d = pd.read_csv(f)
        runs.append(d[d.space == "native"])
    a = pd.concat(runs)
    a["m"] = [names.get((v, int(s))) for v, s in zip(a.variant, a.seed)]
    a = a[a.m.notna()]
    return pd.concat([c, a]).groupby(["m", "organ"]).hdmax_mm.median()


def table_robust():
    fr = pd.read_csv(AB / "stats_fragments.csv")
    la = pd.read_csv(AB / "stats_lateralite.csv").set_index("model")
    hm = hdmax()
    order = [("E3_s42", "E3"), ("A2", "A2"), ("A1", "A1"), ("E2_s42", "E2"), ("A3", "A3")]
    lines = ["| Model | Lungs: median no. of components (L / R) | Lungs: HDmax, mm (L / R) | Patients with > 1 % of the oesophagus outside its largest component | Left/right lung confusion > 5 % |",
             "|---|---|---|---|---|"]
    for m, lab in order:
        f = fr[fr.model == m].set_index("organ")
        lines.append(f"| {lab} | {f.loc['poumon_g','ncc_med']:.0f} / {f.loc['poumon_d','ncc_med']:.0f} | "
                     f"{hm[(m,'poumon_g')]:.1f} / {hm[(m,'poumon_d')]:.1f} | {f.loc['oesophage','pct_frac1']:.1f} % | "
                     f"{int(la.loc[m,'n_sup5'])}/187 ({la.loc[m,'pct']:.1f} %) |")
    return "\n".join(lines)


def table_6oar():
    p = pd.read_csv(P4 / "p4_comparaisons.csv")
    d = pd.read_csv(P4 / "p4_descriptifs.csv").set_index(["m", "organ"])
    lines = ["| OAR | DSC U-Net | DSC SegResNet | Δ DSC (SegResNet − U-Net) | HD95 U-Net (mm) | HD95 SegResNet (mm) | Δ HD95 (mm) |",
             "|---|---|---|---|---|---|---|"]
    for o in ORG + ["moelle", "aorte"]:
        r = p[(p.metrique == "dsc") & (p.organe == o)].iloc[0]
        h = p[(p.metrique == "hd95_mm") & (p.organe == o)].iloc[0]
        dsc_u, dsc_s = d.loc[("U-Net", o), "dsc"], d.loc[("SegResNet", o), "dsc"]
        hd_u, hd_s = d.loc[("U-Net", o), "hd95_mm"].split(" ")[0], d.loc[("SegResNet", o), "hd95_mm"].split(" ")[0]
        lines.append(f"| {EN[o]} | {dsc_u} | {dsc_s} | {ci(r)} | {float(hd_u):.1f} | {float(hd_s):.1f} | {ci(h, 1)} |")
    return "\n".join(lines)


def table_6oar_thorax():
    s = pd.read_csv(ROOT / "phase4_6oar/thorax/stats_thorax.csv")
    cov = pd.read_csv(ROOT / "phase4_6oar/thorax/couverture_thorax.csv").set_index("organ")
    lines = ["| OAR | Reference volume within the lung slices, median [Q1] | Region | DSC U-Net | DSC SegResNet | "
             "Δ DSC (SegResNet − U-Net) | HD95 U-Net (mm) | HD95 SegResNet (mm) | Δ HD95 (mm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for o in ("moelle", "aorte"):
        for region, lab in (("complet", "full extent"), ("thorax", "lung slices")):
            d = s[(s.region == region) & (s.organe == o) & (s.metrique == "dsc")].iloc[0]
            h = s[(s.region == region) & (s.organe == o) & (s.metrique == "hd95_mm")].iloc[0]
            c = cov.loc[o]
            first = f"{EN[o]} | {c.frac_gt_thorax_med:.2f} [{c.frac_gt_thorax_q1:.2f}]" if region == "complet" else " | "
            lines.append(f"| {first} | {lab} | {d['U-Net_med']:.3f} | {d['SegResNet_med']:.3f} | {ci(d)} | "
                         f"{h['U-Net_med']:.1f} | {h['SegResNet_med']:.1f} | {ci(h, 1)} |")
    return "\n".join(lines)


XLEV = {"enc_profond": "Deepest encoder stage", "dec_demi": "Decoder, 1/2 resolution",
        "dec_pre_tete": "Last feature map before the output head", "sortie": "Output logits"}


def _xai_table(path, organs, a, b):
    s = pd.read_csv(path)
    s = s[s.metrique == "in_organ_ratio"]
    lines = ["| Layer level | " + " | ".join(EN[o] for o in organs) + " |", "|---|" + "---|" * len(organs)]
    for lv in XLEV:
        cells = []
        for o in organs:
            r = s[(s.niveau == lv) & (s.organe == o)].iloc[0]
            ns = " ns" if not r.p_holm < 0.05 else ""
            k = 3 if abs(r.diff_mediane) < 0.005 else 2
            cells.append(f"{r[a]:.2f} → {r[b]:.2f} ({r.diff_mediane:+.{k}f}{ns})")
        lines.append(f"| {XLEV[lv]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_xai4():
    return _xai_table(ROOT / "phase3_ablation/xai_matched/analyse/xai_stats_apparies.csv", ORG,
                      "E3_mediane", "E2_mediane")


def table_xai6():
    return _xai_table(ROOT / "phase3_ablation/xai_matched/analyse/xai6_stats_apparies.csv",
                      ORG + ["moelle", "aorte"], "UNet_mediane", "SegResNet_mediane")


def main():
    tpl = (ROOT / "paper/Article_Ablation_6OAR_template.md").read_text(encoding="utf-8")
    out = (tpl.replace("{{TABLE_PHASE2}}", table_phase2()).replace("{{TABLE_PAIRED2}}", table_paired_phase2())
           .replace("{{TABLE_SEEDS}}", table_seeds()).replace("{{TABLE_ABLATION}}", table_ablation())
           .replace("{{TABLE_ROBUST}}", table_robust()).replace("{{TABLE_6OAR}}", table_6oar())
           .replace("{{TABLE_6OAR_THORAX}}", table_6oar_thorax())
           .replace("{{TABLE_XAI4}}", table_xai4()).replace("{{TABLE_XAI6}}", table_xai6()))
    assert "{{" not in out, "placeholder non remplace"
    (ROOT / "paper/Article_Ablation_6OAR_v1.md").write_text(out, encoding="utf-8")
    print("ecrit paper/Article_Ablation_6OAR_v1.md", len(out.split()), "mots")


if __name__ == "__main__":
    main()
