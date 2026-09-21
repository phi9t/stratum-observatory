"""Conservative publication scan. Findings name locations and categories, never secret values.

Public citations and code-owner handles are attribution, not local machine identity.
This is a regression guard, not a guarantee that arbitrary prose contains no PII.
"""
import base64
import gzip
import zlib
import ipaddress
import json
import re
import subprocess
import os
import struct
from pathlib import Path

PATTERNS = {
    'personal-path': re.compile(r'(?:/(?:Users|home|root)/[^\s"\x27<>]+|[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\s"<>]+)'),
    'mac-address': re.compile(r'(?i)(?<![\w:])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![\w:])'),
    'gpu-uuid': re.compile(r'(?i)GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}'),
    'email': re.compile(r'\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'private-key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    'access-token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b'),
    'internal-host': re.compile(r'\b(?:devx_|jumper_)[A-Za-z0-9_-]+'),
}
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]|\.\d)')
IPV6 = re.compile(r'(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]*(?![\w:])')


def findings(path, text):
    found = []
    if path.startswith('evidence/runtime/') or Path(path).name in {'.env', 'id_rsa', 'id_ed25519'}:
        found.append({'path': path, 'line': 0, 'kind': 'private-artifact'})
    for number, line in enumerate(text.splitlines(), 1):
        checked = line.replace('git@github.com:', 'github-ssh:')
        kinds = {kind for kind, pattern in PATTERNS.items() if pattern.search(checked)}
        for pattern in (IPV4, IPV6):
            for match in pattern.finditer(line):
                try:
                    if pattern is IPV4 and re.search(r'(?:MLNX_)?OFED[^\s]*' + re.escape(match.group()), line):
                        continue
                    address = ipaddress.ip_address(match.group())
                except ValueError:
                    continue
                if not address.is_loopback and not address.is_unspecified:
                    kinds.add('ip-address')
        found.extend({'path': path, 'line': number, 'kind': kind} for kind in sorted(kinds))
    return found


def packed_findings(path, text):
    """Inspect our offline gzip payloads as data, including nested fixture payloads."""
    issues = findings(path, text)
    pending = [text]
    seen = set()
    for depth in range(4):
        following = []
        for content in pending:
            for match in re.finditer(r"H4sI[A-Za-z0-9+/=]{24,}", content):
                payload = match.group()
                if payload in seen:
                    continue
                seen.add(payload)
                try:
                    decoded = gzip.decompress(base64.b64decode(payload, validate=True)).decode('utf-8')
                except (ValueError, OSError, EOFError, UnicodeError, zlib.error):
                    continue
                issues.extend(findings(path + ':gzip', decoded))
                following.append(decoded)
        pending = following
        if not pending:
            break
    return issues


def public_files(root):
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=root, check=True, capture_output=True)
    return sorted({p for p in result.stdout.decode().split('\0') if p and (root/p).is_file()})


def scan(root):
    paths = public_files(root)
    issues = []
    for name in paths:
        path = root/name
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
                issues.append({'path': name, 'line': 0, 'kind': 'external-symlink'})
            issues.extend(findings(name, target))
            continue
        data = path.read_bytes()
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            # Pixel streams are compressed random bytes. Review metadata separately.
            offset = 8
            while offset + 12 <= len(data):
                size = struct.unpack('>I', data[offset:offset+4])[0]
                kind = data[offset+4:offset+8]
                if kind not in {b'IHDR', b'IDAT', b'IEND', b'sRGB', b'gAMA', b'cHRM', b'pHYs'}:
                    issues.append({'path': name, 'line': 0, 'kind': 'image-metadata-review-required'})
                offset += size + 12
        else:
            issues.extend(packed_findings(name, data.decode('utf-8', errors='replace')))
    return {'status': 'FAIL' if issues else 'PASS', 'files': len(paths), 'findings': issues,
            'scope': 'Current tracked and non-ignored files only; excludes Git history and ignored private data.'}


def main():
    report = scan(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))
    return int(bool(report['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
