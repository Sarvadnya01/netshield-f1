"""Rewrite artifact roots in mlflow.db from an old repo root to a new one.

Usage:
  python scripts/rewrite_mlflow_paths.py --db mlflow.db --old-root "C:/old/path" --new-root "C:/new/path"
  python scripts/rewrite_mlflow_paths.py --db mlflow.db --old-root "C:/old/path" --new-root "C:/new/path" --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


def rewrite_paths(db_path: str, old_root: str, new_root: str, apply: bool = False) -> int:
    """Rewrite artifact paths in the MLflow SQLite database.

    Args:
        db_path: Path to the mlflow.db file.
        old_root: Old repo root to replace (forward slashes).
        new_root: New repo root to substitute (forward slashes).
        apply: If False (default), dry-run only.

    Returns:
        Number of rows that would be / were updated.
    """
    # Normalize to forward slashes and strip trailing slash
    old_root = old_root.replace("\\", "/").rstrip("/")
    new_root = new_root.replace("\\", "/").rstrip("/")

    if old_root == new_root:
        print("Old and new roots are identical, nothing to do.")
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    total_changes = 0

    # Tables and columns to rewrite
    rewrites = [
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
    ]

    for table, column in rewrites:
        # Check if table/column exists
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            print(f"  {table}.{column}: column not found, skipping")
            continue

        cursor.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} LIKE ?",
            (f"%{old_root}%",),
        )
        count = cursor.fetchone()[0]
        total_changes += count

        if count > 0:
            # Show examples
            cursor.execute(
                f"SELECT {column} FROM {table} WHERE {column} LIKE ? LIMIT 3",
                (f"%{old_root}%",),
            )
            examples = cursor.fetchall()
            print(f"  {table}.{column}: {count} rows to update")
            for (val,) in examples:
                new_val = val.replace(old_root, new_root)
                print(f"    {val}")
                print(f"    -> {new_val}")

            if apply:
                cursor.execute(
                    f"UPDATE {table} SET {column} = REPLACE({column}, ?, ?)",
                    (old_root, new_root),
                )
                print(f"    APPLIED ({count} rows)")
        else:
            print(f"  {table}.{column}: no rows match old_root")

    if apply:
        conn.commit()
        print(f"\nTotal: {total_changes} rows updated.")
    else:
        print(f"\nDRY RUN: {total_changes} rows would be updated. Use --apply to execute.")

    conn.close()
    return total_changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite MLflow artifact paths in SQLite DB",
    )
    parser.add_argument("--db", required=True, help="Path to mlflow.db")
    parser.add_argument("--old-root", required=True, help="Old repo root path")
    parser.add_argument("--new-root", required=True, help="New repo root path")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually apply changes (default is dry-run)",
    )
    args = parser.parse_args()

    print(f"MLflow path rewriter")
    print(f"  DB:       {args.db}")
    print(f"  Old root: {args.old_root}")
    print(f"  New root: {args.new_root}")
    print(f"  Mode:     {'APPLY' if args.apply else 'DRY RUN'}")
    print()

    rewrite_paths(args.db, args.old_root, args.new_root, args.apply)


if __name__ == "__main__":
    main()
