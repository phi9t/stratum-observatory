import tempfile,unittest
from pathlib import Path
from verification import sources
class SourceAcquisitionTests(unittest.TestCase):
    def test_api_exists(self): self.assertTrue(callable(getattr(sources,'acquire',None)))
    def test_never_overwrites_accepted_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'accepted.json';f.write_text('keep')
            with self.assertRaises(ValueError):sources.acquire(root,{}, {},f,f)
            self.assertEqual(f.read_text(),'keep')
    def test_rejects_local_network_before_fetch(self):
        from unittest.mock import Mock
        fetch=Mock()
        with tempfile.TemporaryDirectory() as d:
            result=sources.fetch_source({'kind':'document','url':'http://127.0.0.1/secret','id':'bad'},Path(d),opener=fetch)
        self.assertEqual(result['status'],'FAIL');fetch.assert_not_called()
    def test_missing_revision_does_not_resolve_main(self):
        from unittest.mock import Mock
        fetch=Mock()
        with tempfile.TemporaryDirectory() as d:
            result=sources.fetch_source({'kind':'git','provider':'github','repo':'NVIDIA/nccl','path':'README.md','revision':'main','url':'https://raw.githubusercontent.com/NVIDIA/nccl/main/README.md','id':'bad'},Path(d),opener=fetch)
        self.assertEqual(result['status'],'FAIL');fetch.assert_not_called()
