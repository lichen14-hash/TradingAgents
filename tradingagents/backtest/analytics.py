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
