#!/usr/bin/env python3
"""
Supplementary Figure Generator

Generates all supplementary figures for the manuscript by running the
appropriate scripts in the correct order (respecting dependencies).

Usage:
    python generate_supplementary_figures.py              # Generate all figures
    python generate_supplementary_figures.py --list       # List all figures
    python generate_supplementary_figures.py --fig S1 S3  # Generate specific figures
    python generate_supplementary_figures.py --from S5    # Generate S5 and all after
    python generate_supplementary_figures.py --check      # Check which outputs exist
    python generate_supplementary_figures.py --dry-run    # Show what would be run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from figure_registry import (
    SUPPLEMENTARY_FIGURES,
    get_generation_order,
    RESULT_CODE_DIR,
)


def check_output_exists(fig_id: str) -> bool:
    """Check if the output file for a figure exists."""
    fig = SUPPLEMENTARY_FIGURES.get(fig_id)
    if not fig:
        return False
    output_path = RESULT_CODE_DIR / fig["output"]
    return output_path.exists()


def get_output_mtime(fig_id: str) -> datetime | None:
    """Get modification time of output file."""
    fig = SUPPLEMENTARY_FIGURES.get(fig_id)
    if not fig:
        return None
    output_path = RESULT_CODE_DIR / fig["output"]
    if output_path.exists():
        return datetime.fromtimestamp(output_path.stat().st_mtime)
    return None


def run_figure_script(fig_id: str, dry_run: bool = False) -> bool:
    """
    Run the script to generate a figure.

    Returns True if successful, False otherwise.
    """
    fig = SUPPLEMENTARY_FIGURES.get(fig_id)
    if not fig:
        print(f"  [ERROR] Unknown figure: {fig_id}")
        return False

    if fig["script"] is None:
        print(f"  [SKIP] {fig_id}: No script (pre-generated)")
        return True

    script_path = RESULT_CODE_DIR / fig["script"]

    if not script_path.exists():
        print(f"  [ERROR] {fig_id}: Script not found: {script_path}")
        return False

    print(f"  [RUN] {fig_id}: {fig['title'][:50]}...")
    print(f"        Script: {fig['script']}")

    if dry_run:
        print(f"        (dry-run, not executing)")
        return True

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(RESULT_CODE_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"  [FAIL] {fig_id}: Script returned error code {result.returncode}")
            if result.stderr:
                print(f"        stderr: {result.stderr[:500]}")
            return False

        # Check if output was created
        output_path = RESULT_CODE_DIR / fig["output"]
        if output_path.exists():
            print(f"  [OK] {fig_id}: Output created at {fig['output']}")
            return True
        else:
            print(f"  [WARN] {fig_id}: Script ran but output not found at {fig['output']}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {fig_id}: Script timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  [ERROR] {fig_id}: {e}")
        return False


def list_figures():
    """Print a list of all supplementary figures."""
    print("=" * 80)
    print("SUPPLEMENTARY FIGURES")
    print("=" * 80)

    order = get_generation_order()
    for fig_id in order:
        fig = SUPPLEMENTARY_FIGURES[fig_id]
        exists = check_output_exists(fig_id)
        mtime = get_output_mtime(fig_id)

        status = "[EXISTS]" if exists else "[MISSING]"
        mtime_str = mtime.strftime("%Y-%m-%d %H:%M") if mtime else "N/A"

        deps = ", ".join(fig["dependencies"]) if fig["dependencies"] else "none"

        print(f"\n{fig_id} {status} (modified: {mtime_str})")
        print(f"  Title: {fig['title']}")
        print(f"  Script: {fig['script'] or '(pre-generated)'}")
        print(f"  Output: {fig['output']}")
        print(f"  Dependencies: {deps}")

    print("\n" + "=" * 80)


def check_figures():
    """Check which figure outputs exist."""
    print("=" * 70)
    print("CHECKING SUPPLEMENTARY FIGURE OUTPUTS")
    print("=" * 70)

    missing = []
    existing = []

    for fig_id in get_generation_order():
        fig = SUPPLEMENTARY_FIGURES[fig_id]
        output_path = RESULT_CODE_DIR / fig["output"]

        if output_path.exists():
            mtime = datetime.fromtimestamp(output_path.stat().st_mtime)
            existing.append((fig_id, fig["title"], mtime))
        else:
            missing.append((fig_id, fig["title"], fig["output"]))

    print(f"\nExisting ({len(existing)}):")
    for fig_id, title, mtime in existing:
        print(f"  [OK] {fig_id}: {title[:45]}... ({mtime:%Y-%m-%d %H:%M})")

    if missing:
        print(f"\nMissing ({len(missing)}):")
        for fig_id, title, output in missing:
            print(f"  [MISSING] {fig_id}: {title[:45]}...")
            print(f"            Expected: {output}")

    print("\n" + "=" * 70)
    return len(missing) == 0


def generate_figures(
    figure_ids: list[str] | None = None,
    from_figure: str | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    """
    Generate supplementary figures.

    Args:
        figure_ids: Specific figures to generate (None = all)
        from_figure: Start from this figure (and all after)
        dry_run: Just print what would be done
        force: Regenerate even if output exists
    """
    order = get_generation_order()

    # Determine which figures to generate
    if figure_ids:
        # Specific figures requested
        to_generate = [fid.upper() for fid in figure_ids]
        # Add dependencies
        all_needed = set()
        for fid in to_generate:
            fig = SUPPLEMENTARY_FIGURES.get(fid)
            if fig:
                all_needed.add(fid)
                all_needed.update(fig.get("dependencies", []))
        # Maintain order
        to_generate = [fid for fid in order if fid in all_needed]
    elif from_figure:
        # From specific figure onwards
        from_figure = from_figure.upper()
        if from_figure in order:
            idx = order.index(from_figure)
            to_generate = order[idx:]
        else:
            print(f"[ERROR] Unknown figure: {from_figure}")
            return False
    else:
        # All figures
        to_generate = order

    print("=" * 70)
    print("GENERATING SUPPLEMENTARY FIGURES")
    print("=" * 70)
    print(f"\nFigures to generate: {', '.join(to_generate)}")
    if dry_run:
        print("(DRY RUN - no scripts will be executed)")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for fig_id in to_generate:
        # Check if output exists and skip if not forcing
        if not force and check_output_exists(fig_id):
            mtime = get_output_mtime(fig_id)
            print(f"  [SKIP] {fig_id}: Output exists ({mtime:%Y-%m-%d %H:%M})")
            skip_count += 1
            continue

        if run_figure_script(fig_id, dry_run=dry_run):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Generated: {success_count}")
    print(f"  Skipped (exists): {skip_count}")
    print(f"  Failed: {fail_count}")
    print("=" * 70)

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate supplementary figures for the manuscript"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all supplementary figures"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check which outputs exist"
    )
    parser.add_argument(
        "--fig", "-f",
        nargs="+",
        metavar="ID",
        help="Generate specific figures (e.g., S1 S3 S5)"
    )
    parser.add_argument(
        "--from",
        dest="from_figure",
        metavar="ID",
        help="Generate from this figure onwards (e.g., --from S5)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if output exists"
    )

    args = parser.parse_args()

    if args.list:
        list_figures()
    elif args.check:
        success = check_figures()
        sys.exit(0 if success else 1)
    else:
        success = generate_figures(
            figure_ids=args.fig,
            from_figure=args.from_figure,
            dry_run=args.dry_run,
            force=args.force,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
