"""Generate HTML/PDF report from Skill analysis results.

Usage:
    python scripts/generate_skill_report.py <final_state_json> <bundle_json> [--name NAME]

Example:
    python scripts/generate_skill_report.py skill_output/002602_SZ_final_state.json skill_output/002602.SZ_2026-06-29_latest.json --name "世纪华通"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.datacollector import DataBundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SKILL_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "skill_output"


def generate_report(final_state_path: Path, bundle_path: Path, name: str = "") -> dict:
    """Generate HTML report and optionally PDF from analysis results."""
    # Load data
    with open(final_state_path, encoding="utf-8") as f:
        final_state = json.load(f)
    bundle = DataBundle.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))

    ticker = bundle.metadata.ticker
    if not name:
        name = final_state.get("company_of_interest", ticker)

    # Import the report generation function from run_batch_analysis
    from run_batch_analysis import generate_html_report

    # Skill版直接输出到 skill_output/ 目录，与Web版隔离
    html_path = generate_html_report(ticker, name, final_state, bundle, output_dir=SKILL_OUTPUT_DIR)
    logger.info("HTML report generated: %s", html_path)

    # Try PDF conversion via Chrome headless
    pdf_path = html_path.with_suffix(".pdf")
    chrome = _find_chrome()
    if chrome:
        try:
            subprocess.run(
                [chrome, "--headless", "--disable-gpu",
                 f"--print-to-pdf={pdf_path}", str(html_path)],
                check=True, capture_output=True, timeout=30,
            )
            logger.info("PDF report generated: %s", pdf_path)
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning("PDF generation failed: %s", e)
            pdf_path = None
    else:
        logger.info("Chrome not found, skipping PDF generation")
        pdf_path = None

    result = {
        "status": "success",
        "ticker": ticker,
        "name": name,
        "html_report": str(html_path),
        "pdf_report": str(pdf_path) if pdf_path and pdf_path.exists() else None,
    }
    return result


def _find_chrome() -> str | None:
    """Find Chrome/Chromium executable."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate report from Skill analysis results")
    parser.add_argument("final_state", help="Path to final_state.json")
    parser.add_argument("bundle", help="Path to DataBundle JSON file")
    parser.add_argument("--name", default="", help="Display name for the stock/ETF")
    args = parser.parse_args()

    final_state_path = Path(args.final_state)
    bundle_path = Path(args.bundle)

    if not final_state_path.exists():
        print(f"Error: {final_state_path} not found", file=sys.stderr)
        sys.exit(1)
    if not bundle_path.exists():
        print(f"Error: {bundle_path} not found", file=sys.stderr)
        sys.exit(1)

    result = generate_report(final_state_path, bundle_path, args.name)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
