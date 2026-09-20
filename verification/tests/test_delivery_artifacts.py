import tempfile
import unittest
import json
from pathlib import Path
from verification.engineering import delivery_artifacts
from verification.delivery import artifact_names,page_inventory,guide_inventory

ROOT=Path(__file__).resolve().parents[2]


def fixture(root):
    (root/'src').mkdir()
    (root/'src/pages.mjs').write_bytes((ROOT/'src/pages.mjs').read_bytes())
    (root/'dist').mkdir()
    for name in artifact_names(root):
        (root/'dist'/name).write_text('original '+name)


class DeliveryTests(unittest.TestCase):
    def test_preview_cannot_claim_the_dedicated_page_without_its_identity(self):
        from verification.presentations import judge_preview_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);(root/'previews').mkdir()
            for entry in page_inventory(root):
                for model in entry['models'] or [entry['defaultModel']]:
                    with self.subTest(model=model):
                        (root/'previews/manifest.json').write_text(json.dumps({'captures':[
                            {'name':model,'state':{'model':model,'view':entry.get('view','model')}}]}))
                        self.assertTrue(any(f.status=='FAIL' for f in judge_preview_artifacts(root)))

    def test_dedicated_page_mutation_changes_delivery_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            original=delivery_artifacts(root)
            for entry in page_inventory(root)+guide_inventory(root):
                with self.subTest(page=entry['file']):
                    page=root/'dist'/entry['file'];before=page.read_bytes()
                    page.write_text('modified')
                    changed=delivery_artifacts(root)
                    if page.name!='index.html':self.assertEqual(original['index.html'],changed['index.html'])
                    self.assertNotEqual(original[page.name],changed[page.name])
                    page.write_bytes(before)

    def test_missing_dedicated_artifact_cannot_be_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            for entry in page_inventory(root)+guide_inventory(root):
                with self.subTest(page=entry['file']):
                    page=root/'dist'/entry['file'];before=page.read_bytes();page.unlink()
                    with self.assertRaises(FileNotFoundError):delivery_artifacts(root)
                    page.write_bytes(before)

    def test_guide_preview_must_bind_its_own_artifact(self):
        from verification.presentations import judge_preview_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);(root/'previews').mkdir()
            for entry in guide_inventory(root):
                (root/'previews/manifest.json').write_text(json.dumps({'captures':[
                    {'name':entry['id'],'guide':entry['id'],'artifact':entry['file'],'artifact_sha256':'0'*64}]}))
                self.assertTrue(any(f.status=='FAIL' for f in judge_preview_artifacts(root)))
