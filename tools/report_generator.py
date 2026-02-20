"""Reusable report table writers for analysis outputs.

Designed for direct paper insertion workflows: each table is emitted to both CSV and
Markdown with deterministic float formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


DEFAULT_FLOAT_FMT = "{:.4f}"


def _format_cell(value: object, float_fmt: str) -> str:
    if isinstance(value, float):
        return float_fmt.format(value)
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, float_fmt: str = DEFAULT_FLOAT_FMT) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in df.itertuples(index=False, name=None):
        cells = [_format_cell(v, float_fmt=float_fmt) for v in row]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def write_table_report(
    table: pd.DataFrame,
    output_dir: Path,
    stem: str,
    float_fmt: str = DEFAULT_FLOAT_FMT,
) -> tuple[Path, Path]:
    """Write one table to CSV + Markdown.

    Returns: `(csv_path, markdown_path)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"

    table.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(table, float_fmt=float_fmt), encoding="utf-8")
    return csv_path, md_path


def write_named_table_reports(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    float_fmt: str = DEFAULT_FLOAT_FMT,
) -> dict[str, tuple[Path, Path]]:
    """Write multiple named tables to CSV + Markdown."""
    outputs: dict[str, tuple[Path, Path]] = {}
    for stem, table in tables.items():
        outputs[stem] = write_table_report(
            table=table,
            output_dir=output_dir,
            stem=stem,
            float_fmt=float_fmt,
        )
    return outputs
