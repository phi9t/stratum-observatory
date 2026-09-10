"""The hardware record: read one, check it, and derive the digest evidence may rest on.

`tools/probe-system.py` captures the record on a host that has no copy of this repository.
This module is the repository-side reader: it validates the shape against
`schemas/hardware.schema.json`, re-derives `hardware_sha256` from the attested half with the
house digest, and refuses a fixture where production evidence is expected.

The digest covers `attested` alone. That is the whole point of the split: two probes of the
same unchanged machine agree on it however busy the machine happened to be when each ran.
"""
from pathlib import Path

from .evidence import digest,loads
from .schema import validate as shape_validate

SCHEMA='stratum-hardware/v1'
SCHEMA_PATH=Path(__file__).resolve().parent/'schemas/hardware.schema.json'


def schema():
    return loads(SCHEMA_PATH.read_text())


def hardware_sha256(record):
    """The digest evidence binds to: the attested half, and nothing else."""
    return digest(record['attested'])


def check(record,*,allow_fixtures=False):
    """Return the record, or raise. A record that fails any of these is not evidence."""
    if not isinstance(record,dict):raise ValueError('Hardware record must be an object')
    if record.get('schema')!=SCHEMA:
        raise ValueError(f'Unsupported hardware record version: {record.get("schema")!r}')
    errors=shape_validate(record,schema())
    if errors:raise ValueError('Hardware record does not match its schema: '+'; '.join(errors[:8]))
    if record.get('fixture') and not allow_fixtures:
        raise ValueError('This is a fixture capture and cannot stand as production evidence')
    if record['hardware_sha256']!=hardware_sha256(record):
        raise ValueError('hardware_sha256 does not cover the attested half it is recorded against')
    return record


def load(path,*,allow_fixtures=False):
    return check(loads(Path(path).read_text()),allow_fixtures=allow_fixtures)


# Domains whose runtime obligations attest what a machine or a fabric IS, rather than what our
# own code does. Their receipts are bound to the hardware and to the data model they were
# captured under -- never to the claim manifest or to our function digests, because no edit to
# either can change how much HBM an H100 has. Every other registered domain attests behaviour
# of our own bytes and keeps binding `lock_sha256` and `consumer_sha256`; see
# `verification/runner.py`. This set is closed on purpose: widening it relaxes a real check.
HARDWARE_DOMAINS=frozenset({'gpu','scale_up','scale_out'})


def schema_sha256():
    """Digest of the hardware data model, as a receipt records it in `probe_schema_sha256`.

    Taken over the parsed schema with the house digest, so reformatting the file does not
    retire captures while any change to what the model actually requires does.
    """
    return digest(schema())
