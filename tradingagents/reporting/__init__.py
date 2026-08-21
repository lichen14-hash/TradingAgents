"""Report rendering helpers shared by the batch runner, the web server, and the CLI.

Historically the HTML renderers lived in ``test_output/run_baba_analysis.py`` and
the production batch runner imported them from there. This package is where the
shared pieces belong; the test scripts keep thin wrappers.
"""

from .integrity_report import (
    build_integrity_banner,
    build_stage_rating_table,
    collect_integrity_findings,
    split_data_completeness_findings,
    stage_rating_note,
)

__all__ = [
    "build_integrity_banner",
    "build_stage_rating_table",
    "collect_integrity_findings",
    "split_data_completeness_findings",
    "stage_rating_note",
]
