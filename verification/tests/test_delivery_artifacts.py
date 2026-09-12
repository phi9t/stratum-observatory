import tempfile
import unittest
import json
from pathlib import Path
from verification.engineering import delivery_artifacts


class DeliveryTests(unittest.TestCase):
    def test_preview_cannot_claim_the_dedicated_page_without_its_identity(self):
        from verification.presentations import judge_preview_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'previews').mkdir()
            (root/'previews/manifest.json').write_text(json.dumps({'captures':[
                {'name':'v41','state':{'model':'dsv41'}}]}))
            self.assertTrue(any(f.status=='FAIL' for f in judge_preview_artifacts(root)))

    def test_dedicated_page_mutation_changes_delivery_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'dist').mkdir()
            (root/'dist/index.html').write_text('atlas')
            page=root/'dist/deepseek-v4.1.html';page.write_text('original')
            original=delivery_artifacts(root)
            page.write_text('modified')
            changed=delivery_artifacts(root)
            self.assertEqual(original['index.html'],changed['index.html'])
            self.assertNotEqual(original,changed)

    def test_missing_dedicated_artifact_cannot_be_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'dist').mkdir()
            (root/'dist/index.html').write_text('atlas')
            with self.assertRaises(FileNotFoundError):delivery_artifacts(root)
