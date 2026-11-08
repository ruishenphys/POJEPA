"""交接专用原生捕获：不改命令算法，不重试，不覆盖旧证据，不设置运行时限。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def file_sha(path):
    """流式计算实际文件字节摘要，完整输出不会载入控制台。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    """新目录保存真实子进程退出码；XML失败时只改变捕获器状态，不伪改子进程退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytest-xml", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("-- 后必须有完整命令和参数")
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if args.pytest_xml:
        command += ["--junitxml", str(out/"results.xml")]
    record = dict(argv=command, cwd=str(args.cwd.resolve()), timeout_seconds=None,
        environment={key: os.environ.get(key) for key in ("PYTHONUTF8", "PYTHONWARNINGS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PO_JEPA_RUN_F03_REAL_BATCH",
            "PO_JEPA_F03_FEATURES_ROOT", "PO_JEPA_F03_DEVICE", "CUBLAS_WORKSPACE_CONFIG")},
        xml_applicable=args.pytest_xml, capture_source_sha256=file_sha(Path(__file__)))
    (out/"command-start.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    start = time.monotonic(); code = None
    with (out/"stdout.txt").open("xb") as stdout, (out/"stderr.txt").open("xb") as stderr:
        try:
            code = subprocess.run(command, cwd=args.cwd, stdout=stdout, stderr=stderr).returncode
        except OSError as error:
            record["launch_error"] = repr(error)
    valid_xml = True
    if args.pytest_xml:
        valid_xml = False
        try:
            tree = ET.parse(out/"results.xml").getroot(); cases = list(tree.iter("testcase"))
            counts = dict(tests=len(cases), failures=sum(c.find("failure") is not None for c in cases),
                errors=sum(c.find("error") is not None for c in cases), skipped=sum(c.find("skipped") is not None for c in cases))
            suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
            reported = {key: sum(int(s.get(key, "0")) for s in suites) for key in counts}
            verified = counts == reported
            record.update(xml_counts=counts, xml_reported_counts=reported, counts_verified=verified)
            # 本工具只用于具名真实门槛：skip不能充当验收通过。
            valid_xml = verified and counts["tests"] > 0 and not any(counts[k] for k in ("failures", "errors", "skipped"))
        except (OSError, ET.ParseError, ValueError) as error:
            record.update(counts_verified=False, xml_error=repr(error))
    wrapper_code = (code if code > 0 else 128-code) if code else (0 if code == 0 and valid_xml else 2)
    record.update(exit_code=code, capture_exit_code=wrapper_code, elapsed_seconds=time.monotonic()-start,
        stdout_sha256=file_sha(out/"stdout.txt"), stderr_sha256=file_sha(out/"stderr.txt"),
        xml_sha256=file_sha(out/"results.xml") if (out/"results.xml").exists() else None)
    (out/"exit-code.txt").write_text(str(code) if code is not None else "NOT_STARTED", encoding="utf-8")
    (out/"command.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("exit_code", "capture_exit_code", "xml_applicable")}))
    return wrapper_code


if __name__ == "__main__":
    raise SystemExit(main())
