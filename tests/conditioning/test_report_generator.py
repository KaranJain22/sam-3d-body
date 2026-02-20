from pathlib import Path

import pandas as pd

from tools.report_generator import dataframe_to_markdown, write_table_report


def test_dataframe_to_markdown_formats_table():
    df = pd.DataFrame({"a": [1], "b": [2.34567]})
    md = dataframe_to_markdown(df)

    assert "| a | b |" in md
    assert "2.3457" in md


def test_write_table_report_writes_csv_and_markdown(tmp_path: Path):
    df = pd.DataFrame({"metric": ["mpjpe"], "value": [12.0]})

    csv_path, md_path = write_table_report(df, output_dir=tmp_path, stem="paper_table")

    assert csv_path.exists()
    assert md_path.exists()
    assert "metric,value" in csv_path.read_text(encoding="utf-8")
    assert "| metric | value |" in md_path.read_text(encoding="utf-8")
