"""Methodology references cannot acquire implementation or hardware authority."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from verification import evidence
from verification.runner import required_roles


class MethodologySources(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = b'<p>Methodology equation</p>\n'
        (self.root / 'book.html').write_bytes(self.raw)
        self.source = dict(id='book', kind='document', role='methodology_reference',
                           url='https://jax-ml.github.io/scaling-book/transformers/',
                           version='HTML snapshot 2026-09-12', license='Reference only; no redistribution',
                           cache='book.html', sha256=hashlib.sha256(self.raw).hexdigest())

    def test_allowed_snapshot(self):
        self.assertEqual(evidence.check_source(self.source, self.root), [])

    def test_wrong_origin_and_path(self):
        for url in ('https://jax-ml.github.io/other/transformers/',
                    'https://example.com/scaling-book/transformers/',
                    'https://phi9t.github.io/other/',
                    'https://jax-ml.github.io/scaling-book/../private/',
                    'https://docs.nvidia.com/example',
                    'https://jax-ml.github.io/scaling-book/transformers/#different',
                    'https://jax-ml.github.io/scaling-book/transformers/?edition=other'):
            with self.subTest(url=url):
                self.assertFalse(evidence.valid_url(dict(self.source, url=url)))

    def test_cannot_masquerade_as_code_or_vendor(self):
        for role in ('publisher_code', 'framework_code', 'library_code', 'vendor_document'):
            with self.subTest(role=role):
                self.assertTrue(evidence.check_source(dict(self.source, role=role), self.root))
        git = dict(self.source, kind='git', provider='github', repo='NVIDIA/Megatron-LM',
                   revision='a'*40, path='README.md',
                   url='https://raw.githubusercontent.com/NVIDIA/Megatron-LM/'+'a'*40+'/README.md')
        self.assertFalse(evidence.valid_url(git))
        self.assertFalse(evidence.dual_code_roles([self.source]))
        for domain in ('gpu', 'communication', 'scale_up', 'scale_out'):
            self.assertNotIn('methodology_reference', required_roles({'domain':domain}))

    def test_tampered_snapshot_fails(self):
        (self.root / 'book.html').write_bytes(b'changed equation')
        self.assertIn('SOURCE_HASH', [f.code for f in evidence.check_source(self.source, self.root)])
