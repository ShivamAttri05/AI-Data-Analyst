"""
Gemini API client for AI Data Analyst Mode.
Handles API calls for analysis planning, code generation, and insight synthesis.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai

# ── Load .env if available ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass


# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_MODEL      = "gemini-3-flash-preview"
_MAX_HISTORY_TURNS  = 5      # how many prior turns to include in follow-up context
_CODE_FENCE_RE      = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)
_JSON_ARRAY_RE      = re.compile(r"\[.*?\]", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences so the result is plain Python."""
    return _CODE_FENCE_RE.sub("", text).strip()


def _parse_json_array(text: str) -> Optional[List[str]]:
    """Extract and parse the first JSON array found in *text*."""
    match = _JSON_ARRAY_RE.search(text)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, list):
                return [str(item) for item in value]
        except json.JSONDecodeError:
            pass
    return None


def _format_column_info(column_info: Dict[str, str]) -> str:
    """Render column info dict as a compact bulleted string."""
    return "\n".join(f"  • {col}: {info}" for col, info in column_info.items())


class GeminiClient:
    """
    Client for Google Gemini API.

    All public methods mirror the original interface exactly so existing call
    sites require zero changes.  Improvements are internal: better prompts,
    robust JSON parsing, consistent code-fence stripping, and cleaner errors.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise the Gemini client.

        Args:
            api_key: Gemini API key.  Falls back to the GEMINI_API_KEY
                     environment variable when not supplied.

        Raises:
            ValueError: if no API key can be found.
        """
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. "
                "Pass it to GeminiClient() or set the GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(_DEFAULT_MODEL)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call(self, prompt: str) -> str:
        """
        Make a single Gemini API call and return the stripped text response.

        Raises:
            RuntimeError: wraps any SDK exception with a human-readable message.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    # ── Public interface (identical signatures to original) ────────────────────

    def generate_analysis_steps(
        self,
        user_question: str,
        dataset_summary: str,
        column_info: Dict[str, str],
    ) -> List[str]:
        """
        Break a natural-language question into 3–5 concrete analysis steps.

        Args:
            user_question:   The question the user typed.
            dataset_summary: Pre-formatted dataset overview string.
            column_info:     Mapping of column name → description / dtype.

        Returns:
            List of step strings, e.g. ["Step 1: …", "Step 2: …"].
        """
        prompt = f"""You are a senior data analyst.

A user has uploaded a dataset and asked:
  "{user_question}"

DATASET SUMMARY
{dataset_summary}

COLUMNS
{_format_column_info(column_info)}

TASK
Decompose the question into 3–5 specific, ordered analysis steps that together
give a complete answer.  Each step must be:
  - self-contained and actionable
  - focused on a single operation (filter, aggregate, visualise, correlate, etc.)
  - phrased as an imperative sentence starting with a verb

OUTPUT FORMAT
Return a JSON array of strings and nothing else — no prose, no markdown fences.
Example:
["Calculate monthly sales totals by region", "Plot a bar chart comparing regions", "Identify the top 3 performing months"]
"""
        try:
            raw = self._call(prompt)
            steps = _parse_json_array(raw)
            if steps:
                return steps
            # Fallback: treat non-empty lines as steps
            return [ln.strip() for ln in raw.splitlines() if ln.strip()]
        except RuntimeError as exc:
            return [f"Could not generate steps: {exc}"]

    def generate_analysis_code(
        self,
        analysis_step: str,
        dataset_summary: str,
        column_info: Dict[str, str],
        df_shape: tuple,
    ) -> str:
        """
        Generate executable Python code for one analysis step.

        Args:
            analysis_step:   The step description from generate_analysis_steps().
            dataset_summary: Pre-formatted dataset overview string.
            column_info:     Mapping of column name → description / dtype.
            df_shape:        (n_rows, n_cols) tuple from df.shape.

        Returns:
            Runnable Python source code as a plain string (no fences).
        """
        prompt = f"""You are an expert Python data analyst.

TASK
Write Python code that performs this single analysis step:
  "{analysis_step}"

CONTEXT
  • The DataFrame is already available as the variable `df`.
  • Shape: {df_shape[0]:,} rows × {df_shape[1]} columns

DATASET SUMMARY
{dataset_summary}

COLUMNS
{_format_column_info(column_info)}

CODE REQUIREMENTS
  1. Use only `df` — do NOT re-read or redefine it.
  2. Use pandas for all data operations.
  3. If a chart is appropriate, create it with matplotlib / seaborn.
     - Apply a clean style: `plt.style.use('seaborn-v0_8-whitegrid')`
     - Include a descriptive title and axis labels.
     - Call `plt.tight_layout()` before the last `plt.show()`.
  4. Print a concise summary of the key numeric findings (3 lines max).
  5. Handle potential errors gracefully (missing columns, NaN values).
  6. The code must be complete and runnable with no placeholders.

OUTPUT
Return ONLY the Python code — no markdown fences, no commentary.
"""
        try:
            raw = self._call(prompt)
            return _strip_code_fences(raw)
        except RuntimeError as exc:
            return f"# Code generation failed: {exc}"

    def generate_insights(
        self,
        analysis_results: str,
        user_question: str,
        dataset_summary: str,
    ) -> str:
        """
        Synthesise plain-English insights from analysis output.

        Args:
            analysis_results: Captured stdout / text output from the analysis steps.
            user_question:    The original user question.
            dataset_summary:  Pre-formatted dataset overview string.

        Returns:
            Markdown-formatted insight text ready to display in the UI.
        """
        prompt = f"""You are a senior data analyst presenting findings to a business stakeholder.

ORIGINAL QUESTION
  "{user_question}"

DATASET SUMMARY
{dataset_summary}

ANALYSIS OUTPUT
{analysis_results}

TASK
Write a concise, actionable response that directly answers the question above.
Structure your response with these three sections using bold headers:

**Key Findings**
2–4 bullet points summarising the most important numbers or patterns.

**What This Means**
1–2 sentences translating the data into plain business language.

**Recommended Actions**
2–3 concrete, prioritised actions the reader should take next.

Keep the total response under 250 words.  Be specific — cite numbers from the
analysis output wherever possible.  Do not hedge; give clear conclusions.
"""
        try:
            return self._call(prompt)
        except RuntimeError as exc:
            return f"Could not generate insights: {exc}"

    def generate_followup_response(
        self,
        user_question: str,
        chat_history: List[Dict[str, str]],
        dataset_summary: str,
        column_info: Dict[str, str],
    ) -> str:
        """
        Generate Python code for a follow-up question, aware of prior context.

        Args:
            user_question: The new question from the user.
            chat_history:  List of {"user": …, "assistant": …} dicts.
            dataset_summary: Pre-formatted dataset overview string.
            column_info:     Mapping of column name → description / dtype.

        Returns:
            Runnable Python source code as a plain string (no fences).
        """
        # Format the most recent turns for context
        recent = chat_history[-_MAX_HISTORY_TURNS:]
        history_str = "\n\n".join(
            f"User: {turn['user']}\nAnalyst: {turn['assistant']}"
            for turn in recent
        )

        prompt = f"""You are an expert Python data analyst continuing an ongoing analysis session.

CONVERSATION HISTORY (most recent {len(recent)} turn(s))
{history_str}

NEW QUESTION
  "{user_question}"

DATASET SUMMARY
{dataset_summary}

COLUMNS
{_format_column_info(column_info)}

TASK
Write Python code that answers the new question, taking the previous analysis
into account where relevant.

CODE REQUIREMENTS
  1. Use `df` as the existing DataFrame — do NOT redefine it.
  2. If the new question builds on a result from the history, reference that
     result explicitly in a comment.
  3. Produce a visualisation with matplotlib / seaborn when helpful.
  4. Print a 1–3 line numeric summary of the result.
  5. The code must be complete and runnable with no placeholders.

OUTPUT
Return ONLY the Python code — no markdown fences, no commentary.
"""
        try:
            raw = self._call(prompt)
            return _strip_code_fences(raw)
        except RuntimeError as exc:
            return f"# Code generation failed: {exc}"


# ── Factory function (identical signature to original) ────────────────────────

def get_gemini_client(api_key: Optional[str] = None) -> GeminiClient:
    """
    Create and return a GeminiClient instance.

    Args:
        api_key: Optional API key override.  When omitted, the client reads
                 from the GEMINI_API_KEY environment variable.

    Returns:
        Configured GeminiClient ready to use.
    """
    return GeminiClient(api_key)