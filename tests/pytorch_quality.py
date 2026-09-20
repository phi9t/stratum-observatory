"""Contracts of the Python programs users actually download."""
import ast
import typing
import unittest
from torch import Tensor
from pytorch_test_support import records


class PythonProgramQualityTests(unittest.TestCase):
    def test_each_record_is_independent_and_all_parameters_are_typed(self):
        for record in records():
            with self.subTest(example=record['id']):
                namespace={'__name__':'stratum_download'}
                exec(compile(record['code'],record['id'],'exec'),namespace)
                tree=ast.parse(record['code'])
                for node in ast.walk(tree):
                    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        self.assertIsNotNone(node.returns,node.name)
                        for argument in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:
                            if argument.arg not in {'self','cls'}:
                                self.assertIsNotNone(argument.annotation,f'{node.name}.{argument.arg}')
                    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                        self.assertNotRegex(node.name,r'^(BB_|bb_|Readable|readable_)')

    def test_tensor_annotations_explain_their_axes(self):
        def check(annotation,where):
            if typing.get_origin(annotation) is typing.Annotated:
                tensor,*metadata=typing.get_args(annotation)
                self.assertIs(tensor,Tensor,where)
                self.assertTrue(any(isinstance(x,str) and x.strip() for x in metadata),where)
                return
            self.assertIsNot(annotation,Tensor,f'{where}: bare Tensor has no named axes')
            for member in typing.get_args(annotation):
                check(member,where)
        for record in records():
            namespace={'__name__':'stratum_download'}
            exec(compile(record['code'],record['id'],'exec'),namespace)
            for node in ast.parse(record['code']).body:
                if isinstance(node,ast.FunctionDef):
                    functions=[namespace[node.name]]
                elif isinstance(node,ast.ClassDef):
                    functions=[getattr(namespace[node.name],m.name) for m in node.body
                               if isinstance(m,ast.FunctionDef)]
                else:
                    continue
                for function in functions:
                    for name,hint in typing.get_type_hints(function,globalns=namespace,include_extras=True).items():
                        check(hint,f'{record["id"]}:{function.__name__}.{name}')


if __name__=='__main__':
    unittest.main()
