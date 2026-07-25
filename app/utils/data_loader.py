"""
Data loading utilities for AI Data Analyst Mode.
Handles CSV, Excel, and JSON file loading with robust error handling,
encoding detection, and rich dataset profiling.
"""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
_CSV_ENCODINGS   = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]
_CSV_SEPARATORS  = [",", ";", "\t", "|"]
_HIGH_CARDINALITY_THRESHOLD = 15   # show sample values below this unique count
_SAMPLE_VALUES_MAX          = 8    # cap on enumerated unique values shown


class DataLoader:
    """
    Loads, validates, and profiles tabular datasets from uploaded files.

    Supports CSV (auto-detects encoding and delimiter), Excel (.xlsx / .xls),
    and JSON (records or columns orientation).
    """

    SUPPORTED_FORMATS: Dict[str, str] = {
        ".csv":  "csv",
        ".xlsx": "excel",
        ".xls":  "excel",
        ".json": "json",
    }

    # ── Loading ────────────────────────────────────────────────────────────────

    @staticmethod
    def load_data(file) -> Tuple[pd.DataFrame, str]:
        """
        Load a dataset from a Streamlit uploaded-file object.

        Tries multiple encodings for CSV files and falls back gracefully.
        Resets the file pointer before each read attempt so the same object
        can be passed multiple times without error.

        Args:
            file: ``st.uploaded_file_manager.UploadedFile`` instance.

        Returns:
            ``(DataFrame, format_type)`` where *format_type* is one of
            ``"csv"``, ``"excel"``, or ``"json"``.

        Raises:
            ValueError: if the extension is unsupported or the file cannot
                        be parsed by any attempted strategy.
        """
        ext = Path(file.name).suffix.lower()

        if ext not in DataLoader.SUPPORTED_FORMATS:
            supported = ", ".join(DataLoader.SUPPORTED_FORMATS)
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Accepted formats: {supported}"
            )

        fmt = DataLoader.SUPPORTED_FORMATS[ext]

        try:
            if fmt == "csv":
                df = DataLoader._load_csv(file)
            elif fmt == "excel":
                df = DataLoader._load_excel(file)
            else:
                df = DataLoader._load_json(file)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not parse '{file.name}': {exc}") from exc

        if df.empty:
            raise ValueError("The file loaded successfully but contains no rows.")

        # Attempt cheap dtype inference after load
        df = DataLoader._infer_dtypes(df)

        return df, fmt

    @staticmethod
    def _load_csv(file) -> pd.DataFrame:
        """Try every (encoding, separator) combination until one succeeds."""
        raw = file.read()

        for encoding in _CSV_ENCODINGS:
            for sep in _CSV_SEPARATORS:
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw),
                        encoding=encoding,
                        sep=sep,
                        engine="python",       # tolerates unusual separators
                        on_bad_lines="warn",
                    )
                    # Require at least 2 columns — a 1-col result usually means
                    # the wrong delimiter was chosen
                    if df.shape[1] >= 2:
                        return df
                except Exception:
                    continue

        raise ValueError(
            "Could not parse CSV with any supported encoding "
            f"({', '.join(_CSV_ENCODINGS)}) or delimiter "
            f"({', '.join(repr(s) for s in _CSV_SEPARATORS)})."
        )

    @staticmethod
    def _load_excel(file) -> pd.DataFrame:
        """Load the first sheet of an Excel workbook."""
        raw = file.read()
        return pd.read_excel(io.BytesIO(raw), sheet_name=0)

    @staticmethod
    def _load_json(file) -> pd.DataFrame:
        """Load JSON — tries records then columns orientation."""
        raw = file.read()
        for orient in ("records", "columns", "index", "split"):
            try:
                return pd.read_json(io.BytesIO(raw), orient=orient)
            except Exception:
                continue
        raise ValueError("JSON file could not be parsed in any recognised orientation.")

    @staticmethod
    def _infer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to downcast object columns to numeric or datetime where safe.
        Does not modify the original; returns a new DataFrame.
        """
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            # Try numeric first
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df), 1) > 0.9:
                df[col] = converted
                continue
            # Try datetime
            try:
                dt = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if dt.notna().sum() / max(len(df), 1) > 0.9:
                    df[col] = dt
            except Exception:
                pass
        return df

    # ── Profiling ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build a comprehensive profile of a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary with keys:
            ``shape``, ``rows``, ``columns``, ``column_names``,
            ``dtypes``, ``missing_values``, ``missing_percentages``,
            ``numeric_columns``, ``categorical_columns``,
            ``datetime_columns``, ``numeric_stats`` (if any),
            ``sample_rows``, ``memory_mb``, ``duplicate_rows``.
        """
        n_rows = len(df)

        missing_counts = df.isnull().sum()
        missing_pcts   = (missing_counts / n_rows * 100).round(2) if n_rows else missing_counts * 0

        numeric_cols  = df.select_dtypes(include="number").columns.tolist()
        cat_cols      = df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols       = df.select_dtypes(include="datetime").columns.tolist()

        summary: Dict[str, Any] = {
            "shape":               df.shape,
            "rows":                n_rows,
            "columns":             df.shape[1],
            "column_names":        df.columns.tolist(),
            "dtypes":              df.dtypes.astype(str).to_dict(),
            "missing_values":      missing_counts.to_dict(),
            "missing_percentages": missing_pcts.to_dict(),
            "numeric_columns":     numeric_cols,
            "categorical_columns": cat_cols,
            "datetime_columns":    dt_cols,
            "memory_mb":           round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
            "duplicate_rows":      int(df.duplicated().sum()),
            "sample_rows":         df.head(5).to_dict(orient="records"),
        }

        if numeric_cols:
            summary["numeric_stats"] = (
                df[numeric_cols]
                .describe()
                .round(4)
                .to_dict()
            )

        return summary

    @staticmethod
    def format_summary_for_ai(summary: Dict[str, Any]) -> str:
        """
        Serialise a summary dict into a compact, LLM-friendly plain-text block.

        Args:
            summary: Dictionary produced by :meth:`get_dataset_summary`.

        Returns:
            Multi-line string ready to paste into a prompt.
        """
        lines: List[str] = [
            f"Shape: {summary['rows']:,} rows × {summary['columns']} columns",
            f"Memory: {summary['memory_mb']} MB",
            f"Duplicate rows: {summary['duplicate_rows']:,}",
            "",
            "Columns:",
        ]

        for col in summary["column_names"]:
            dtype       = summary["dtypes"].get(col, "unknown")
            n_missing   = summary["missing_values"].get(col, 0)
            pct_missing = summary["missing_percentages"].get(col, 0.0)
            miss_str    = f", {n_missing:,} missing ({pct_missing:.1f}%)" if n_missing else ""
            lines.append(f"  • {col}  [{dtype}]{miss_str}")

        # Column group overview
        def _join_or_none(lst: List[str]) -> str:
            return ", ".join(lst) if lst else "none"

        lines += [
            "",
            f"Numeric columns    : {_join_or_none(summary['numeric_columns'])}",
            f"Categorical columns: {_join_or_none(summary['categorical_columns'])}",
            f"Datetime columns   : {_join_or_none(summary['datetime_columns'])}",
        ]

        # Numeric statistics — compact format
        if "numeric_stats" in summary:
            lines.append("")
            lines.append("Numeric statistics (mean / std / min / max):")
            for col, stats in summary["numeric_stats"].items():
                mean = stats.get("mean", 0)
                std  = stats.get("std",  0)
                lo   = stats.get("min",  0)
                hi   = stats.get("max",  0)
                lines.append(f"  • {col}: {mean:.3g} / {std:.3g} / {lo:.3g} / {hi:.3g}")

        return "\n".join(lines)

    @staticmethod
    def get_column_info(df: pd.DataFrame) -> Dict[str, str]:
        """
        Build a per-column description string suitable for LLM context.

        For low-cardinality columns (< threshold), lists the actual unique
        values so the model knows the valid categories.  For high-cardinality
        columns, shows the count of unique values and a short sample instead.

        Args:
            df: Input DataFrame.

        Returns:
            ``{column_name: description_string}`` for every column.
        """
        info: Dict[str, str] = {}
        n_rows = max(len(df), 1)

        for col in df.columns:
            dtype       = str(df[col].dtype)
            n_unique    = df[col].nunique(dropna=True)
            n_missing   = int(df[col].isnull().sum())
            pct_missing = n_missing / n_rows * 100

            parts = [dtype]

            if n_unique <= _HIGH_CARDINALITY_THRESHOLD:
                sample = df[col].dropna().unique().tolist()[:_SAMPLE_VALUES_MAX]
                parts.append(f"values: {sample}")
            else:
                # Show a short random sample for flavour
                sample = (
                    df[col]
                    .dropna()
                    .sample(min(_SAMPLE_VALUES_MAX, n_unique), random_state=0)
                    .tolist()
                )
                parts.append(f"{n_unique:,} unique (e.g. {sample})")

            if n_missing:
                parts.append(f"{n_missing:,} missing ({pct_missing:.1f}%)")

            info[col] = ", ".join(parts)

        return info