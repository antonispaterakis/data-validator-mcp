"""
Quick sanity check — 12-row synthetic dataset, 3 labels, 3 intentional mislabels.

Run from project root:
    python sanity_check.py
"""

import csv
import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
ROWS = [
    # cardiology (6 clean)
    ("Patient presents with chest pain radiating to the left arm and shortness of breath.", "cardiology"),
    ("ECG shows ST elevation consistent with myocardial infarction.", "cardiology"),
    ("Heart palpitations and irregular rhythm detected on Holter monitor.", "cardiology"),
    ("Echocardiogram reveals reduced ejection fraction, consistent with heart failure.", "cardiology"),
    ("Troponin levels elevated, indicating acute coronary syndrome.", "cardiology"),
    ("Hypertension managed with beta-blockers and ACE inhibitors.", "cardiology"),
    # orthopedics (4 clean)
    ("X-ray confirms displaced fracture of the distal radius.", "orthopedics"),
    ("Patient complains of knee joint swelling and reduced range of motion.", "orthopedics"),
    ("Bone density scan reveals osteoporosis with high fracture risk.", "orthopedics"),
    ("Post-operative rehabilitation following total hip replacement.", "orthopedics"),
    # --- intentional mislabels ---
    # row 10: clearly cardiology text, wrong label
    ("Aortic valve stenosis detected; patient referred for valve replacement surgery.", "orthopedics"),
    # row 11: clearly orthopedics text, wrong label
    ("Comminuted fracture of the femur shaft; surgical fixation recommended.", "cardiology"),
]

# ---------------------------------------------------------------------------
# Write CSV to a temp file
# ---------------------------------------------------------------------------
def create_temp_csv(rows):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.writer(tmp)
    writer.writerow(["text", "label"])
    writer.writerows(rows)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
def main():
    # Add project root so the src package resolves with relative imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from src.pipeline import ValidationPipeline

    csv_path = create_temp_csv(ROWS)
    print(f"\n[sanity] CSV written to: {csv_path}")
    print(f"[sanity] {len(ROWS)} rows — expected mislabels at rows 10 and 11\n")

    pipeline = ValidationPipeline(
        k_neighbors=5,
        agreement_threshold=0.75,
        llm_model="meta-llama-3.1-8b-instruct",
    )

    summary = pipeline.run(csv_path, text_col="text", label_col="label")

    os.unlink(csv_path)

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total rows         : {summary['total_rows']}")
    print(f"  Clusters found     : {summary['n_clusters']}")
    print(f"  Flagged by cluster : {summary['n_flagged_by_clustering']}")
    print(f"  Blob rows judged   : {summary['n_blob_rows_judged']}")
    print(f"  LLM bad            : {summary['llm_bad']}")
    print(f"  LLM good           : {summary['llm_good']}")
    print(f"  LLM unknown        : {summary['llm_unknown']}")
    print(f"  Est. mislabel %    : {summary['estimated_mislabel_pct']}%")

    print("\n" + "=" * 60)
    print("TOKEN STATS")
    print("=" * 60)
    ts = summary["token_stats"]
    print(f"  rows_scanned_by_llm   : {ts['rows_scanned_by_llm']} / {ts['total_rows']} total")
    print(f"  tokens_input          : {ts['tokens_input']}")
    print(f"  tokens_output         : {ts['tokens_output']}")
    print(f"  tokens_used           : {ts['tokens_used']}")
    print(f"  tokens_if_brute_force : {ts['tokens_if_brute_force']}")
    print(f"  efficiency_ratio      : {ts['efficiency_ratio']}x  ← pipeline vs brute-force")

    print("\n" + "=" * 60)
    print("FLAGGED ROWS")
    print("=" * 60)
    for r in pipeline.flagged_rows:
        nr = " [NEEDS REVIEW]" if r.get("needs_review") else ""
        print(
            f"  row {r['row_index']:>2} | {r['assigned_label']:<12} → "
            f"{r['suggested_label'] or '?':<12} | "
            f"{r['llm_verdict']:<7} {r['llm_confidence']:<6} | "
            f"{r['detection_source']}{nr}"
        )
        print(f"           {r['text_preview'][:80]}…")

    print("\n" + "=" * 60)
    print("CLUSTER OVERVIEW")
    print("=" * 60)
    for c in pipeline.cluster_info:
        print(
            f"  cluster {c['cluster_id']} | size={c['size']} | "
            f"dominant={c['dominant_label']} | dist={c['label_distribution']}"
        )

    print()


if __name__ == "__main__":
    main()
