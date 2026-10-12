"""Preserve the existing public interfaces and CUDA implementation sources."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads(Path(__file__).with_name("fixtures").joinpath("public_api.json").read_text())


def signature_tree(value):
    if value is Ellipsis:
        return {"constant": "Ellipsis"}
    if isinstance(value, ast.AST):
        return {
            "node": type(value).__name__,
            **{
                name: signature_tree(item)
                for name, item in ast.iter_fields(value)
                if item is not None and item != []
            },
        }
    if isinstance(value, list):
        return [signature_tree(item) for item in value]
    return value


def signature_hash(value):
    canonical = json.dumps(signature_tree(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_public_exports_and_signatures_are_unchanged():
    tree = ast.parse(ROOT.joinpath("dmuon/__init__.py").read_text(encoding="utf-8"))
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert exports == CONTRACT["public"]
    for name, contract in CONTRACT["entries"].items():
        tree = ast.parse(ROOT.joinpath(contract["path"]).read_text(encoding="utf-8"))
        if contract["kind"] == "constant":
            node = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == name for target in node.targets
                )
            )
            assert signature_hash(node.value) == contract["value"], name
            continue
        node = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == contract.get("name", name)
        )
        if contract["kind"] == "class":
            constructor = next(
                (
                    method
                    for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__"
                ),
                None,
            )
            actual = signature_hash(constructor.args) if constructor else None
            fields = [
                signature_hash(field) for field in node.body if isinstance(field, ast.AnnAssign)
            ]
            assert fields == contract["fields"], name
        else:
            actual = signature_hash(node.args)
            returns = signature_hash(node.returns) if node.returns else None
            assert returns == contract["returns"], name
        assert actual == contract["args"], name


def test_cuda_specific_kernel_sources_are_unchanged():
    for path, expected in CONTRACT["cuda_sources"].items():
        # Git checkouts may normalize line endings; that does not change code.
        content = ROOT.joinpath(path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected, path
