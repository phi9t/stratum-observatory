"""Versioned, deliberately small verification result vocabulary."""
from dataclasses import asdict, dataclass
from typing import Any

STATUSES = {'PASS', 'FAIL', 'BLOCKED', 'NOT_RUN', 'NOT_APPLICABLE'}

@dataclass(frozen=True)
class Finding:
    code: str
    domain: str
    status: str
    subject: str
    message: str
    remedy: str = ''
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError('Unknown finding status')

    def json(self) -> dict[str, Any]:
        return asdict(self)

def aggregate(findings: list[Finding]) -> str:
    statuses = {x.status for x in findings}
    if not statuses:
        return 'BLOCKED'
    for status in ('FAIL', 'BLOCKED', 'NOT_RUN'):
        if status in statuses:
            return status
    return 'PASS' if 'PASS' in statuses else 'NOT_APPLICABLE'

def exit_code(status: str) -> int:
    return 0 if status == 'PASS' else 1 if status == 'FAIL' else 2
