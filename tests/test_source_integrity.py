"""Small static checks for defects that can silently remove test coverage."""

import ast
from pathlib import Path


def source_problems(text, filename):
    tree = ast.parse(text, filename=filename)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = set()
            for child in node.body:
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name in names:
                        problems.append(f"{filename}:{child.lineno}: duplicate definition {child.name}")
                    names.add(child.name)
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            ended = False
            for child in value:
                if not isinstance(child, ast.stmt):
                    continue
                if ended:
                    problems.append(f"{filename}:{child.lineno}: unreachable statement")
                ended = isinstance(child, (ast.Return, ast.Raise, ast.Break, ast.Continue))
    return problems


def test_public_source_has_no_shadowed_definitions_or_unreachable_statements():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/rulelens").glob("*.py")) + sorted((root / "tests").glob("*.py"))
    problems = [problem for path in paths for problem in source_problems(path.read_text(encoding="utf-8"), str(path))]
    assert not problems, "\n".join(problems)


def test_static_check_catches_shadowing_and_unreachable_code():
    text = "def example():\n    return 1\n    return 2\ndef example():\n    pass\n"
    problems = source_problems(text, "example.py")
    assert any("duplicate definition" in problem for problem in problems)
    assert any("unreachable statement" in problem for problem in problems)


def test_public_modules_do_not_import_private_helpers():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src/rulelens").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("internal"), str(path)
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("internal") for alias in node.names), str(path)
