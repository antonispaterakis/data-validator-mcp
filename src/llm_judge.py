"""
LLM-as-judge: uses a local Ollama model to confirm whether a flagged row is truly mislabeled.

Communicates with Ollama via its OpenAI-compatible API (http://localhost:11434/v1).
No API key or internet connection required — all inference runs locally.

Two judge modes:
  judge_row()      — used for clustering-flagged rows. Has strong cluster context
                     (dominant label + example texts). High signal.
  judge_blob_row() — used for rows in low-purity "blob" clusters where no dominant
                     label exists. Relies on the full label taxonomy + cluster
                     distribution instead. Also asks the LLM to suggest the correct
                     label directly, since cluster dominant is unreliable here.

Key efficiency insight from Theocharopoulos et al. (2025): the LLM is only
invoked on the subset of rows that the clustering step flagged as suspicious —
not on the entire dataset. The blob pass extends this to mixed clusters while
still skipping rows in clean, pure clusters.
"""

from __future__ import annotations

import json

from openai import OpenAI

_JUDGE_PROMPT = """\
You are a data quality expert reviewing a machine learning training dataset.

A text sample has been flagged as potentially mislabeled. It was grouped by \
semantic similarity with other texts whose dominant label is different from \
the label assigned to it.

--- FLAGGED SAMPLE ---
Text: {text}
Assigned label: "{label}"

--- CLUSTER CONTEXT ---
Dominant label in this semantic cluster: "{cluster_label}"
Other texts in the same cluster (for reference):
{cluster_examples}
--- END CONTEXT ---

Decide whether the assigned label "{label}" is CORRECT for this text.

Reply ONLY with valid JSON in exactly this format — no extra keys, no markdown:
{{
  "verdict": "good",
  "confidence": "high",
  "reasoning": "one or two sentences"
}}

"verdict" must be "good" (label is correct) or "bad" (label is wrong).
"confidence" must be "high", "medium", or "low".\
"""

_BLOB_JUDGE_PROMPT = """\
You are a data quality expert reviewing a machine learning training dataset.

This text sample belongs to a semantically mixed cluster — labels are distributed \
across multiple categories with no clear dominant label. Automatic detection could \
not determine if this sample is correctly labeled, so your judgment is required.

--- SAMPLE ---
Text: {text}
Assigned label: "{label}"

--- CONTEXT ---
All valid labels in this dataset: {all_labels}
Label distribution in this sample's cluster: {cluster_distribution}
--- END CONTEXT ---

Decide whether the assigned label "{label}" is CORRECT for this text.
If it is wrong, also provide the correct label from the valid labels list.

Reply ONLY with valid JSON in exactly this format — no extra keys, no markdown:
{{
  "verdict": "good",
  "confidence": "high",
  "reasoning": "one or two sentences",
  "suggested_label": null
}}

"verdict" must be "good" (label is correct) or "bad" (label is wrong).
"confidence" must be "high", "medium", or "low".
"suggested_label" must be one of the valid labels if verdict is "bad", otherwise null.\
"""


class LLMJudge:
    """Thin wrapper around the Ollama OpenAI-compatible API for single-row label judgement."""

    def __init__(self, model: str = "llama3.2"):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Required by the SDK, ignored by Ollama
        )
        self.model = model

    def judge_row(
        self,
        text: str,
        label: str,
        cluster_dominant: str,
        cluster_examples: list[str],
    ) -> dict:
        """
        First-pass judge: called for rows flagged by clustering.
        Has strong cluster context — dominant label + example texts.

        Returns: verdict, confidence, reasoning.
        """
        examples_str = "\n".join(
            f"  - {ex[:140]}" for ex in cluster_examples[:5]
        ) or "  (no other examples available)"

        prompt = _JUDGE_PROMPT.format(
            text=text[:600],
            label=label,
            cluster_label=cluster_dominant,
            cluster_examples=examples_str,
        )
        return self._call(prompt, expect_suggested_label=False)

    def judge_blob_row(
        self,
        text: str,
        label: str,
        all_labels: list[str],
        cluster_distribution: dict[str, int],
    ) -> dict:
        """
        Second-pass judge: called for rows in low-purity blob clusters.
        No dominant label to reference — uses full taxonomy + cluster distribution.

        Returns: verdict, confidence, reasoning, suggested_label.
        """
        dist_str = ", ".join(f"{lbl}×{cnt}" for lbl, cnt in sorted(
            cluster_distribution.items(), key=lambda x: -x[1]
        ))
        prompt = _BLOB_JUDGE_PROMPT.format(
            text=text[:600],
            label=label,
            all_labels=", ".join(f'"{l}"' for l in all_labels),
            cluster_distribution=dist_str,
        )
        return self._call(prompt, expect_suggested_label=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str, expect_suggested_label: bool) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            result = json.loads(raw)
            out = {
                "verdict": str(result.get("verdict", "unknown")).lower(),
                "confidence": str(result.get("confidence", "low")).lower(),
                "reasoning": str(result.get("reasoning", raw[:200])),
            }
            if expect_suggested_label:
                sl = result.get("suggested_label")
                out["suggested_label"] = str(sl) if sl and sl != "null" else None
            return out
        except json.JSONDecodeError:
            out = {
                "verdict": "unknown",
                "confidence": "low",
                "reasoning": raw[:300],
            }
            if expect_suggested_label:
                out["suggested_label"] = None
            return out
