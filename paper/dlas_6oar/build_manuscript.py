# -*- coding: utf-8 -*-
"""Builds paper/dlas_6oar/Manuscript_DLAS_6OAR.md (+ references file) from the template, the tables computed by
analysis_dlas.py and references.py. Citations [@key; @key2] are numbered by first appearance (Vancouver).

    .venv\\Scripts\\python.exe paper/dlas_6oar/build_manuscript.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import REFS  # noqa: E402

TAB = HERE / "tables"
EN = {"poumon_g": "Left lung", "poumon_d": "Right lung", "coeur": "Heart", "oesophage": "Oesophagus",
      "moelle": "Spinal cord", "aorte": "Aorta"}


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def t6():
    o = pd.read_csv(TAB / "outliers_SegResNet.csv")
    o = o[o.organ != "oesophage"].copy()
    o["OAR"] = o.organ.map(EN)
    o["Patient"] = o.patient_id.str.replace("PATIENT_", "")
    o["DSC"] = o.dsc.map(lambda x: f"{x:.3f}")
    o["HD95 (mm)"] = o.hd95_mm.map(lambda x: f"{x:.1f}")
    o["Volume error"] = o.vol_rel_err.map(lambda x: f"{x:+.0%}")
    o["Sequence / study"] = o.scanning_sequence.astype(str).str.replace("unkown", "unknown") + " / " + \
        o.study_type.astype(str).str.replace("unkown", "unknown").str.replace("mri ", "")
    o["Probable cause"] = o.probable_cause.str.replace("unkown", "unknown")
    return md_table(o[["OAR", "Patient", "DSC", "HD95 (mm)", "Volume error", "Sequence / study", "Probable cause"]])


def main():
    txt = (HERE / "manuscript_template.md").read_text(encoding="utf-8")
    t4 = pd.read_csv(TAB / "T4_geometric_SegResNet.csv").drop(columns=["n", "Empty predictions"])
    t4 = t4.fillna("—")
    rep = {"{{T1}}": md_table(pd.read_csv(TAB / "T1_cohort.csv")),
           "{{T2}}": md_table(pd.read_csv(TAB / "T2_contours.csv")),
           "{{T4}}": md_table(t4),
           "{{T4B}}": md_table(pd.read_csv(TAB / "T4b_baseline_UNet.csv")),
           "{{T6}}": t6()}
    for k, v in rep.items():
        txt = txt.replace(k, v)

    order = []
    def num(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        ns = []
        for k in keys:
            if k not in REFS:
                raise KeyError(f"reference inconnue : {k}")
            if k not in order:
                order.append(k)
            ns.append(order.index(k) + 1)
        ns = sorted(set(ns))
        out, i = [], 0
        while i < len(ns):                       # compress ranges 3,4,5 -> 3–5
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            out.append(f"{ns[i]}–{ns[j]}" if j - i >= 2 else ", ".join(map(str, ns[i:j + 1])))
            i = j + 1
        return "[" + ", ".join(out) + "]"
    body, _, tail = txt.partition("## References")
    body = re.sub(r"\[(@[^\]]+)\]", num, body)
    tail = re.sub(r"\[(@[^\]]+)\]", num, tail)
    reflist = "\n".join(f"{i}. {REFS[k][0]}" for i, k in enumerate(order, 1))
    out = body + "## References" + tail.replace("{{REFERENCES}}", reflist)
    assert "{{" not in out and "[@" not in out
    (HERE / "Manuscript_DLAS_6OAR.md").write_text(out, encoding="utf-8")

    unused = sorted(set(REFS) - set(order))
    unverified = [k for k in order if not REFS[k][1]]
    (HERE / "References_DLAS_6OAR.md").write_text(
        "# References (Vancouver, numbered in order of first citation)\n\n"
        "Verified entries: retrieved from PubMed on 06/11/2026 (DOI as returned by PubMed). "
        "Entries flagged [TO VERIFY] are non-PubMed sources whose DOI/pages were not confirmed.\n\n"
        + reflist + "\n", encoding="utf-8")
    abstract = out.split("## Abstract")[1].split("*Keywords")[0]
    n_abs = len(re.findall(r"[A-Za-z0-9][\w.\-–/%]*", abstract.replace("**", "")))
    title_words = len(out.splitlines()[0].lstrip("# ").split())
    print(f"references cited: {len(order)} | unused: {unused} | unverified cited: {unverified}")
    print(f"abstract ~{n_abs} words | title {title_words} words | manuscript {len(out.split())} words")


if __name__ == "__main__":
    main()
