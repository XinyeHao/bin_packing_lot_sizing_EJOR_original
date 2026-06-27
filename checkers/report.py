"""Checker 报告结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Violation:
    checker: str
    code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class CheckReport:
    name: str
    passed: bool = True
    violations: list[Violation] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add_violation(self, checker: str, code: str, message: str, **details) -> None:
        self.passed = False
        self.violations.append(Violation(checker, code, message, details))

    def merge(self, other: CheckReport) -> None:
        if not other.passed:
            self.passed = False
        self.violations.extend(other.violations)
        self.metrics.update(other.metrics)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.name}"]
        for v in self.violations:
            lines.append(f"  - [{v.checker}/{v.code}] {v.message}")
        for key, val in self.metrics.items():
            lines.append(f"  * {key}: {val}")
        return "\n".join(lines)
