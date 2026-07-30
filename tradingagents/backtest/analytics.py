"""Aggregate analytics queries for the backtest dashboard."""

from __future__ import annotations

from .db import BacktestDB


class BacktestAnalytics:
    def __init__(self, db: BacktestDB):
        self.db = db

    def summary(self) -> dict:
        conn = self.db.get_connection()
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome_date IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                SUM(CASE WHEN outcome_date IS NULL THEN 1 ELSE 0 END) as pending,
                AVG(CASE WHEN outcome_date IS NOT NULL THEN direction_correct END) as accuracy,
                AVG(CASE WHEN outcome_date IS NOT NULL THEN alpha_return END) as avg_alpha
            FROM predictions"""
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "resolved": row["resolved"] or 0,
            "pending": row["pending"] or 0,
            "accuracy_pct": round((row["accuracy"] or 0) * 100, 1),
            "avg_alpha_pct": round((row["avg_alpha"] or 0) * 100, 2),
        }

    def accuracy_by_rating(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT
                rating,
                signal_numeric,
                COUNT(*) as total,
                SUM(direction_correct) as correct,
                ROUND(AVG(direction_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(raw_return) * 100, 2) as avg_return_pct,
                ROUND(AVG(alpha_return) * 100, 2) as avg_alpha_pct
            FROM predictions
            WHERE outcome_date IS NOT NULL
            GROUP BY rating
            ORDER BY signal_numeric DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def accuracy_by_ticker(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT
                ticker,
                MAX(name) as name,
                COUNT(*) as total,
                SUM(direction_correct) as correct,
                ROUND(AVG(direction_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(alpha_return) * 100, 2) as avg_alpha_pct
            FROM predictions
            WHERE outcome_date IS NOT NULL
            GROUP BY ticker
            ORDER BY total DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def accuracy_timeline(self, window: int = 20) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT trade_date, direction_correct
            FROM predictions
            WHERE outcome_date IS NOT NULL
            ORDER BY trade_date, run_timestamp"""
        ).fetchall()
        if not rows:
            return []
        results = []
        data = [dict(r) for r in rows]
        for i in range(window - 1, len(data)):
            chunk = data[i - window + 1: i + 1]
            correct = sum(1 for c in chunk if c["direction_correct"])
            results.append({
                "date": chunk[-1]["trade_date"],
                "accuracy_pct": round(correct / window * 100, 1),
                "count": window,
            })
        return results

    def debate_analysis(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT
                do.winning_side,
                COUNT(*) as total,
                SUM(p.direction_correct) as correct,
                ROUND(AVG(p.direction_correct) * 100, 1) as accuracy_pct
            FROM debate_outcomes do
            JOIN predictions p ON do.prediction_id = p.id
            WHERE p.outcome_date IS NOT NULL AND do.debate_type = 'investment'
            GROUP BY do.winning_side"""
        ).fetchall()
        return [dict(r) for r in rows]

    def session_comparison(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT
                session,
                COUNT(*) as total,
                SUM(direction_correct) as correct,
                ROUND(AVG(direction_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(alpha_return) * 100, 2) as avg_alpha_pct
            FROM predictions
            WHERE outcome_date IS NOT NULL
            GROUP BY session"""
        ).fetchall()
        return [dict(r) for r in rows]

    def feedback_comparison(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT
                feedback_enabled,
                COUNT(*) as total,
                SUM(direction_correct) as correct,
                ROUND(AVG(direction_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(alpha_return) * 100, 2) as avg_alpha_pct
            FROM predictions
            WHERE outcome_date IS NOT NULL
            GROUP BY feedback_enabled"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Full evaluation for one-click optimization
    # ------------------------------------------------------------------

    def full_evaluation(self) -> dict:
        """Comprehensive evaluation report for LLM optimization analysis."""
        return {
            "summary": self.summary(),
            "by_rating": self.accuracy_by_rating(),
            "by_ticker": self.accuracy_by_ticker(),
            "by_session": self.session_comparison(),
            "by_debate": self.debate_analysis(),
            "by_feedback": self.feedback_comparison(),
            "timeline": self.accuracy_timeline(window=10),
            "worst_predictions": self._worst_predictions(limit=10),
            "best_predictions": self._best_predictions(limit=5),
            "signal_bias": self._signal_bias(),
        }

    def _worst_predictions(self, limit: int = 10) -> list[dict]:
        """Get predictions with worst alpha returns."""
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT ticker, name, trade_date, rating, session,
                   raw_return, alpha_return, executive_summary
            FROM predictions
            WHERE outcome_date IS NOT NULL AND alpha_return IS NOT NULL
            ORDER BY alpha_return ASC
            LIMIT ?""",
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["raw_return_pct"] = round((d.pop("raw_return") or 0) * 100, 2)
            d["alpha_return_pct"] = round((d.pop("alpha_return") or 0) * 100, 2)
            # Truncate summary for LLM context
            summary = d.get("executive_summary") or ""
            d["executive_summary"] = summary[:200]
            results.append(d)
        return results

    def _best_predictions(self, limit: int = 5) -> list[dict]:
        """Get predictions with best alpha returns."""
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT ticker, name, trade_date, rating, session,
                   raw_return, alpha_return, executive_summary
            FROM predictions
            WHERE outcome_date IS NOT NULL AND alpha_return IS NOT NULL
            ORDER BY alpha_return DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["raw_return_pct"] = round((d.pop("raw_return") or 0) * 100, 2)
            d["alpha_return_pct"] = round((d.pop("alpha_return") or 0) * 100, 2)
            summary = d.get("executive_summary") or ""
            d["executive_summary"] = summary[:200]
            results.append(d)
        return results

    def _signal_bias(self) -> dict:
        """Detect signal distribution bias."""
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT rating, COUNT(*) as count
            FROM predictions
            GROUP BY rating
            ORDER BY count DESC"""
        ).fetchall()
        if not rows:
            return {"distribution": [], "is_biased": False, "dominant_signal": None}
        data = [dict(r) for r in rows]
        total = sum(d["count"] for d in data)
        for d in data:
            d["pct"] = round(d["count"] / total * 100, 1) if total else 0
        # Biased if any single signal > 50%
        dominant = data[0]
        is_biased = dominant["pct"] > 50
        return {
            "distribution": data,
            "is_biased": is_biased,
            "dominant_signal": dominant["rating"] if is_biased else None,
            "dominant_pct": dominant["pct"],
        }
