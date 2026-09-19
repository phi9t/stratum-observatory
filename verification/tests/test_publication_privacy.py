"""Publication must exclude private captures and machine-specific report paths."""
import tempfile
import unittest
from pathlib import Path

from verification import privacy
from verification.engineering import public_argv, public_log


class PrivacyTests(unittest.TestCase):
    def test_detects_identifiers_without_echoing_them(self):
        samples = {
            'home path': '/' + 'Users/example/project',
            'IPv4': '.'.join(['10', '23', '45', '67']),
            'IPv6': ':'.join(['2001', 'db8', '1234', '', '42']),
            'MAC': ':'.join(['aa', 'bb', 'cc', 'dd', 'ee', 'ff']),
            'GPU identity': 'GPU-' + '12345678-1234-1234-1234-123456789abc',
            'email': 'person' + '@' + 'example.org',
            'private key': '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----',
        }
        for kind, value in samples.items():
            with self.subTest(kind=kind):
                found = privacy.findings('notes.txt', value)
                self.assertTrue(found)
                self.assertNotIn(value, str(found))

    def test_versions_and_loopback_are_not_machine_identity(self):
        self.assertEqual(privacy.findings('notes.txt', 'Python 3.13.12; Chromium 140.0.7339.16; http://127.0.0.1:8000'), [])

    def test_private_capture_is_rejected_even_without_recognized_text(self):
        self.assertTrue(privacy.findings('evidence/runtime/device/capture.bin', 'opaque'))

    def test_argv_preserves_arguments_without_publishing_interpreter_path(self):
        command = [str(Path.home()/'venv/bin/python3'), '-m', 'unittest']
        self.assertEqual(public_argv(command), ['python3', '-m', 'unittest'])
        self.assertEqual(command[0], str(Path.home()/'venv/bin/python3'))

    def test_log_paths_are_removed_without_changing_test_results(self):
        root = Path.home()/'workspace/project'
        self.assertEqual(public_log(f'{root}/test.py: FAIL; {Path.home()}/cache', root),
                         '<repo>/test.py: FAIL; <home>/cache')

    def test_archive_excludes_history_and_ignored_files_and_normalizes_ownership(self):
        import importlib.util
        import subprocess
        import tarfile
        spec = importlib.util.spec_from_file_location('public_export', Path(__file__).resolve().parents[2]/'tools/export-public.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root/'.gitignore').write_text('private/\n.publication/\n')
            (root/'README.md').write_text('Public source')
            (root/'private').mkdir()
            (root/'private/capture').write_text('/' + 'Users/example/private')
            destination = root/'.publication/source.tar.gz'
            module.export(root, destination)
            first = destination.read_bytes()
            module.export(root, destination)
            self.assertEqual(first, destination.read_bytes())
            with tarfile.open(destination) as archive:
                self.assertEqual(set(archive.getnames()), {'stratum/.gitignore', 'stratum/README.md'})
                for member in archive.getmembers():
                    self.assertEqual((member.uid, member.gid, member.uname, member.gname, member.mtime), (0, 0, '', '', 0))
            (root/'accidental.txt').write_text('/' + 'Users/example/private')
            with self.assertRaises(ValueError):
                module.export(root, destination)
            self.assertEqual(first, destination.read_bytes())

    def test_current_public_files_are_clean(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(privacy.scan(root)['findings'], [])
