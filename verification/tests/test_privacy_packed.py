import base64
import gzip
import unittest
from verification.privacy import packed_findings

class PackedPrivacyTests(unittest.TestCase):
    def test_nested_offline_payload_cannot_hide_machine_identity(self):
        payload = '/' + 'Users' + '/example-user/project'
        for _ in range(2):
            payload = '"' + base64.b64encode(gzip.compress(payload.encode(), mtime=0)).decode() + '"'
        findings = packed_findings('portable.html', payload)
        self.assertEqual([row['kind'] for row in findings], ['personal-path'])
    def test_corrupt_or_benign_payload_is_not_executed(self):
        self.assertEqual(packed_findings('portable.html', 'H4sI' + 'a'*40), [])
        self.assertEqual(packed_findings('portable.html', base64.b64encode(gzip.compress(b'const model = "public";', mtime=0)).decode()), [])
