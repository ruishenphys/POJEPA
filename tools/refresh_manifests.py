"""v7 打包前运行：按当前文件重算 SOURCE_MAP.json 的候选摘要与 MANIFEST.json。

v4–v6 的交付校验（workflow.verify_bundle、tests/test_delivery.py）要求每个源码文件的
SHA 与登记一致。v7 有意修改了源码，所以这里把改动过的文件登记为新候选，同时把旧值
保存在 v7_previous_candidate_sha256，便于追溯。不改动任何源码。

用法：python tools/refresh_manifests.py
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {"results", "features", "__pycache__", ".git", ".pytest_cache", "calibration"}
MANIFEST_SUFFIXES = {".py", ".yaml", ".md", ".json", ".toml", ".tex"}
# 运行时会被 calibrate 改写的配置不进清单（与 v4–v6 相同）。
MUTABLE = {"configs/train/v4.yaml", "configs/train/v7_calibration.yaml"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WithoutDocstrings(ast.NodeTransformer):
    """与 tests/test_delivery.py 相同：去掉文档字符串后比较可执行语法树。"""
    def visit_Module(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
        return node
    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Module


def ast_sha(path: Path) -> str:
    tree = WithoutDocstrings().visit(ast.parse(path.read_text(encoding="utf-8")))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def refresh_source_map() -> list[str]:
    path = ROOT/"SOURCE_MAP.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    changed = []
    for row in source["files"]:
        target = ROOT/row["target"]
        if "candidate_sha256" in row:
            now = sha(target)
            if now != row["candidate_sha256"]:
                row.setdefault("v7_previous_candidate_sha256", row["candidate_sha256"])
                row["candidate_sha256"] = now
                changed.append(row["target"])
        if "executable_ast_sha256" in row:
            row["executable_ast_sha256"] = ast_sha(target)
        if row.get("operation") == "identical_bytes" and not row["target"].endswith("/__init__.py"):
            if sha(target) != row["sha256"]:
                row["operation"] = "v7_modified"
                row["candidate_sha256"] = sha(target)
                changed.append(row["target"])
    source.setdefault("v7_changes", {})["files_changed_since_v6"] = sorted(set(
        changed+source.get("v7_changes", {}).get("files_changed_since_v6", [])))
    path.write_text(json.dumps(source, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return changed


def refresh_manifest() -> int:
    path = ROOT/"MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = {}
    for item in sorted(ROOT.rglob("*")):
        rel = item.relative_to(ROOT)
        if (not item.is_file() or rel.parts[0] in EXCLUDE_DIRS or any(p in EXCLUDE_DIRS for p in rel.parts)
                or item.suffix not in MANIFEST_SUFFIXES or rel.as_posix() in MUTABLE|{"MANIFEST.json"}):
            continue
        files[rel.as_posix()] = sha(item)
    manifest["files_sha256"] = files
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return len(files)


if __name__ == "__main__":
    changed = refresh_source_map()
    count = refresh_manifest()
    print(f"SOURCE_MAP：{len(changed)} 个文件登记为新候选；MANIFEST：{count} 个文件")
