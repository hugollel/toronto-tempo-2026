"""
02_load_db.py — Load raw files into a DuckDB database
=====================================================
Creates data/wnba.duckdb and loads every raw file into the "raw" schema,
one table per dataset (all seasons stacked). Nothing is cleaned here.

It also produces a schema drift report: columns that exist in some
seasons but not in others (the source changes its structure over time).

Run from the project folder:
    python3 pipeline/02_load_db.py
"""

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
DB_FILE = PROJECT_DIR / "data" / "wnba.duckdb"
REPORT_FILE = PROJECT_DIR / "data" / "schema_report.csv"

DATASETS = ["schedule", "team_box", "player_box", "pbp"]


def main() -> None:
    con = duckdb.connect(str(DB_FILE))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    report = []

    for name in DATASETS:
        files = sorted((RAW_DIR / name).glob("*.parquet"))
        if not files:
            print(f"-- {name}: no files found, skipped")
            continue

        # 1. Load all seasons into one table.
        #    union_by_name: columns are matched by name, a column missing in
        #    one season is filled with NULL instead of crashing.
        #    filename: keeps track of which file each row came from.
        con.execute(f"""
            CREATE OR REPLACE TABLE raw.{name} AS
            SELECT * FROM read_parquet(
                '{RAW_DIR / name}/*.parquet',
                union_by_name = true,
                filename = true
            )
        """)

        # 2. Row counts per season (quick sanity check).
        counts = con.execute(f"""
            SELECT season, COUNT(*) AS n_rows
            FROM raw.{name}
            GROUP BY season
            ORDER BY season
        """).fetchall()
        counts_txt = ", ".join(f"{s}: {n}" for s, n in counts)
        print(f"\nOK  raw.{name}  ({counts_txt})")

        # 3. Schema drift: which columns exist in which file.
        columns_by_file = {}
        for f in files:
            cols = con.execute(
                f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{f}'))"
            ).fetchall()
            columns_by_file[f.stem] = {c[0] for c in cols}

        all_cols = set().union(*columns_by_file.values())
        common = set.intersection(*columns_by_file.values())
        drift = sorted(all_cols - common)

        for col in sorted(all_cols):
            report.append({
                "dataset": name,
                "column": col,
                **{stem: (col in cols) for stem, cols in columns_by_file.items()},
            })

        if drift:
            print("    Schema drift (column -> files that have it):")
            for col in drift:
                present = [s for s, cols in columns_by_file.items() if col in cols]
                print(f"      {col}: {', '.join(present)}")

        # 4. Column list with types, so we can write the cleaning step.
        n_cols = len(con.execute(f"DESCRIBE raw.{name}").fetchall())
        print(f"    {n_cols} columns (full list in the schema report)")

    pd.DataFrame(report).to_csv(REPORT_FILE, index=False)
    con.close()
    print(f"\nDatabase: {DB_FILE.relative_to(PROJECT_DIR)}")
    print(f"Schema report: {REPORT_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
