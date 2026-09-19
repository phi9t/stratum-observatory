#!/usr/bin/env python3
"""Export current reviewed files without historical or local machine metadata."""
import gzip
import io
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verification.privacy import scan, public_files


def export(root, destination):
    report = scan(root)
    if report['findings']:
        raise ValueError('Privacy scan failed; run npm run verify:privacy for locations')
    paths = public_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    try:
        with temporary.open('wb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name in paths:
                    path = root/name
                    # Materialize safe internal links so the archive has no link traversal.
                    data = path.read_bytes()
                    info = tarfile.TarInfo('stratum/'+name)
                    info.size = len(data)
                    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                    archive.addfile(info, io.BytesIO(data))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(paths)


if __name__ == '__main__':
    count = export(ROOT, ROOT/'.publication/stratum-public.tar.gz')
    print(f'Exported {count} files to .publication/stratum-public.tar.gz; no Git history or local owner metadata.')
