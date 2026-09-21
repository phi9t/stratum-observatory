"""Chapter symbol links resolve in their own pinned sources, without executing them."""
import ast
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def defined_symbols(raw):
    result = set()
    def walk(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                result.add(prefix + node.name)
                walk(node.body, prefix + node.name + '.')
    walk(ast.parse(raw).body)
    return result


class MechanismSourceTests(unittest.TestCase):
    def test_symbol_ownership_does_not_accept_a_similar_class(self):
        symbols = defined_symbols('class Other:\n def forward(self): pass\n')
        self.assertNotIn('Attention.forward', symbols)
        self.assertIn('Other.forward', symbols)

    def test_chapter_symbols_resolve_in_the_cited_pinned_files(self):
        catalog = {s['id']: s for s in json.loads((ROOT/'verification/manifests/sources.json').read_text())['sources']}
        script = "import {mechanismChapter,MECHANISM_STAGES} from './src/mechanism-chapters.mjs'; import {MODELS} from './src/models.mjs'; console.log(JSON.stringify(MECHANISM_STAGES.flatMap(s=>mechanismChapter(MODELS.dsv41,s).sections)));"
        sections = json.loads(subprocess.check_output(['node','--input-type=module','-e',script],cwd=ROOT,text=True))
        for section in sections:
            symbols = set()
            for source_id in section['sources']:
                source = catalog[source_id]
                path = ROOT/source['cache']
                if not path.exists():
                    self.skipTest('Pinned source cache unavailable; source-symbol resolution NOT_RUN')
                try:
                    symbols.update(defined_symbols(path.read_text()))
                except (SyntaxError, UnicodeError):
                    pass  # Configurations and reports contain no Python definitions.
            with self.subTest(section=section['id']):
                self.assertEqual(set(section['symbols']) - symbols, set())
