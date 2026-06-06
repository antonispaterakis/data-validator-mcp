"""
LLM-as-judge: uses a local LM Studio model to confirm whether a flagged row is truly mislabeled.

Communicates with LM Studio via its OpenAI-compatible API (http://localhost:1234/v1).
No API key or internet connection required — all inference runs locally.

Two judge modes:
  judge_row()      — Pass 1: clustering-flagged rows. Strong context: anchor example
                     for the dominant label + cluster peer texts.
  judge_blob_row() — Pass 2: rows in low-purity blob clusters. Uses K=3 nearest
                     neighbors from pure clusters (relative context) + per-label
                     anchors (absolute context). Also asks the LLM to suggest the
                     correct label directly.

Token tracking:
  Both modes accumulate input + output token counts from the response
  (via response.usage.prompt_tokens / completion_tokens). Access via
  judge.token_stats after the run.

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
semantic similarity with other texts whose dominant label differs from its \
assigned label.

--- ANCHOR (verified representative example for label "{cluster_label}") ---
{anchor_text}
--- END ANCHOR ---

--- FLAGGED SAMPLE ---
Text: {text}
Assigned label: "{label}"

--- CLUSTER CONTEXT ---
Dominant label in this semantic cluster: "{cluster_label}"
Other texts in the same cluster (for reference):
{cluster_examples}
--- END CONTEXT ---

Compare the flagged sample against the anchor example above. Decide whether \
the assigned label "{label}" is CORRECT for this text.

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

This text sample belongs to a semantically mixed cluster — no dominant label \
could be determined automatically. Your judgment is required.

--- NEAREST NEIGHBOR CONTEXT (confirmed examples from high-purity clusters) ---
{neighbor_texts}
--- END NEIGHBORS ---

--- ANCHOR EXAMPLES (verified representative example per candidate label) ---
{anchor_texts}
--- END ANCHORS ---

--- SAMPLE ---
Text: {text}
Assigned label: "{label}"

--- CONTEXT ---
All valid labels in this dataset: {all_labels}
Label distribution in this sample's cluster: {cluster_distribution}
--- END CONTEXT ---

Compare the sample against the neighbor context and anchor examples above. \
Decide whether the assigned label "{label}" is CORRECT for this text. \
If it is wrong, provide the correct label from the valid labels list.

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


_KNN_JUDGE_PROMPT = """\
You are a data quality expert reviewing a machine learning training dataset.

A text sample is flagged as potentially mislabeled: only {n_matching} of its {k} \
nearest semantic neighbours share its assigned label ({agreement:.0%} agreement).

--- ANCHOR (verified representative example for label "{label}") ---
{anchor_text}
--- END ANCHOR ---

--- NEAREST NEIGHBOURS (top 5 most semantically similar texts and their labels) ---
{neighbor_texts}
--- END NEIGHBOURS ---

--- FLAGGED SAMPLE ---
Text: {text}
Assigned label: "{label}"
--- END SAMPLE ---

All valid labels in this dataset: {all_labels}

Compare the sample against the anchor and neighbours. Does the assigned label \
"{label}" seem correct for this text? If not, which label from the valid labels \
list fits better?

Reply ONLY with valid JSON, no markdown:
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

    def __init__(self, model: str = "meta-llama-3.1-8b-instruct"):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",  # Required by the SDK, ignored by LM Studio
        )
        self.model = model
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    @property
    def token_stats(self) -> dict:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def judge_row(
        self,
        text: str,
        label: str,
        cluster_dominant: str,
        cluster_examples: list[str],
        anchor_text: str = "",
    ) -> dict:
        """
        Pass 1 judge: called for rows flagged by clustering.
        Context: anchor example for the dominant label + cluster peer texts.

        Returns: verdict, confidence, reasoning.
        """
        examples_str = "\n".join(
            f"  - {ex[:140]}" for ex in cluster_examples[:5]
        ) or "  (no other examples available)"

        prompt = _JUDGE_PROMPT.format(
            text=text[:600],
            label=label,
            cluster_label=cluster_dominant,
            anchor_text=anchor_text[:300] if anchor_text else "(not available)",
            cluster_examples=examples_str,
        )
        return self._call(prompt, expect_suggested_label=False)

    def judge_blob_row(
        self,
        text: str,
        label: str,
        all_labels: list[str],
        cluster_distribution: dict[str, int],
        neighbor_contexts: list[dict] | None = None,
        anchor_texts: dict[str, str] | None = None,
    ) -> dict:
        """
        Pass 2 judge: called for rows in low-purity blob clusters.
        Context: K=3 nearest neighbors from pure clusters + per-label anchors.

        Returns: verdict, confidence, reasoning, suggested_label.
        """
        dist_str = ", ".join(f"{lbl}×{cnt}" for lbl, cnt in sorted(
            cluster_distribution.items(), key=lambda x: -x[1]
        ))

        # Format neighbor context
        if neighbor_contexts:
            neighbor_lines = "\n".join(
                f'  [{nc["label"]}] {nc["text"][:150]}'
                for nc in neighbor_contexts
            )
        else:
            neighbor_lines = "  (no neighbor context available)"

        # Format anchor texts — only show anchors for candidate labels in this cluster
        candidate_labels = list(cluster_distribution.keys()) or all_labels
        if anchor_texts:
            anchor_lines = "\n".join(
                f'  [{lbl}]: {anchor_texts[lbl][:200]}'
                for lbl in candidate_labels
                if lbl in anchor_texts
            ) or "  (not available)"
        else:
            anchor_lines = "  (not available)"

        prompt = _BLOB_JUDGE_PROMPT.format(
            text=text[:600],
            label=label,
            all_labels=", ".join(f'"{l}"' for l in all_labels),
            cluster_distribution=dist_str,
            neighbor_texts=neighbor_lines,
            anchor_texts=anchor_lines,
        )
        return self._call(prompt, expect_suggested_label=True)

    def judge_knn_row(
        self,
        text: str,
        label: str,
        anchor_text: str,
        neighbor_contexts: list[dict],
        agreement: float,
        n_matching: int,
        k: int,
        all_labels: list[str],
    ) -> dict:
        """
        Unified KNN judge: context = anchor for assigned label + K nearest neighbours.
        Returns: verdict, confidence, reasoning, suggested_label.
        """
        neighbor_lines = "\n".join(
            f'  [{nc["label"]}] {nc["text"][:150]}'
            for nc in neighbor_contexts[:5]
        ) or "  (no neighbour context available)"

        prompt = _KNN_JUDGE_PROMPT.format(
            text=text[:600],
            label=label,
            anchor_text=anchor_text[:300] if anchor_text else "(not available)",
            neighbor_texts=neighbor_lines,
            agreement=agreement,
            n_matching=n_matching,
            k=k,
            all_labels=", ".join(f'"{l}"' for l in all_labels),
        )
        return self._call(prompt, expect_suggested_label=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str, expect_suggested_label: bool) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        # Accumulate token counts from LM Studio response
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens or 0
            self._total_output_tokens += response.usage.completion_tokens or 0

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
