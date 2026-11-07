"""
Facts store: every number an agent may quote is computed by code and registered here with an id.
Agents cite facts ([F3], [T1]) or use placeholders ({{F3}}); the critic checks answers against this store.
"""
import math
import re
from typing import Any, Dict, List, Optional, Tuple


def fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:.4g}"
    return str(value)


class FactStore:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        data = data or {}
        self.facts: List[Dict[str, Any]] = list(data.get("facts", []))
        self.tables: List[Dict[str, Any]] = list(data.get("tables", []))

    def add(self, label: str, value: Any, source: str = "engine") -> str:
        fid = f"F{len(self.facts) + 1}"
        self.facts.append({"id": fid, "label": label, "value": value, "source": source})
        return fid

    def add_table(self, title: str, columns: List[str], rows: List[List[Any]], source: str = "sql") -> str:
        tid = f"T{len(self.tables) + 1}"
        self.tables.append({"id": tid, "title": title, "columns": columns, "rows": rows, "source": source})
        return tid

    def table(self, tid: str) -> Optional[Dict[str, Any]]:
        return next((t for t in self.tables if t["id"] == tid), None)

    def fact(self, fid: str) -> Optional[Dict[str, Any]]:
        return next((f for f in self.facts if f["id"] == fid), None)

    def numbers(self) -> List[float]:
        nums: List[float] = []
        for f in self.facts:
            if isinstance(f["value"], (int, float)) and not isinstance(f["value"], bool):
                nums.append(float(f["value"]))
        for t in self.tables:
            for row in t["rows"]:
                for cell in row:
                    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                        nums.append(float(cell))
        return nums

    def render(self, text: str) -> Tuple[str, List[str]]:
        """Replaces {{F3}} placeholders with formatted values. Returns (text, unknown_ids)."""
        unknown: List[str] = []

        def sub(m: re.Match) -> str:
            f = self.fact(m.group(1))
            if not f:
                unknown.append(m.group(1))
                return "[?]"
            return fmt(f["value"])

        return re.sub(r"\{\{\s*(F\d+)\s*\}\}", sub, text), unknown

    def prompt_block(self, limit: int = 120) -> str:
        lines = [f"{f['id']}: {f['label']} = {fmt(f['value'])}" for f in self.facts[:limit]]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"facts": self.facts, "tables": self.tables}
