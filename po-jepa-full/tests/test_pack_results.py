"""tools/pack_results.py: every packed file is rebuilt byte for byte, repeated blocks are stored once."""

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    """Imported under its real name, so that worker processes started with spawn (Windows) can import it."""
    if str(ROOT/"tools") not in sys.path:
        sys.path.insert(0, str(ROOT/"tools"))
    import pack_results
    assert Path(pack_results.__file__).resolve() == ROOT/"tools/pack_results.py"
    return pack_results


def _repo_json_bytes(value):
    """Same encoder as runtime cache._json_bytes (the writer of report/updates/trajectories files)."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str)+"\n").encode()


PROVENANCE = dict(encoder=dict(config={f"layer_{i}": dict(width=1408, heads=16, sha256=hashlib.sha256(str(i).encode()).hexdigest())
                                       for i in range(40)}), checkpoint_sha256="ab"*32, note="编码器来源")


def _tree(base: Path):
    """A small results tree: repeated provenance, several writers' formats, and files that must stay raw."""
    run = base/"results"/"scene"/"job"
    (run/"updates").mkdir(parents=True)
    (run/"trajectories").mkdir()
    (run/"rows").mkdir()
    for step in range(1, 41):
        record = dict(completed_updates=step, losses=dict(jepa=1/step, time=0.1*step, total=-0.0),
                      provenance=PROVENANCE, queries=[dict(trajectory_id=step, start_tick=5, duration_cycles=2.0)])
        (run/"updates"/f"{step:04d}.json").write_bytes(_repo_json_bytes(record))
    records = [dict(index=i, values=dict(mse=i*1e-7+0.123456789012345), source=PROVENANCE) for i in range(12)]
    (run/"trajectories"/"0001.json").write_bytes(_repo_json_bytes(dict(records=records, summary=dict(n=12))))
    for i, r in enumerate(records):
        (run/"rows"/f"{i:05d}.json").write_bytes(_repo_json_bytes(r))
    (run/"curves.json").write_text(json.dumps(dict(loss=[1.0, float("nan"), 0.5], name="曲线"), indent=1))
    (run/"summary.json").write_text(json.dumps(dict(name="汇总", rows=list(range(300))), indent=1, ensure_ascii=False))
    (run/"compact.json").write_text(json.dumps(dict(a=[PROVENANCE]*3), separators=(",", ":")))
    (run/"spaced.json").write_text(json.dumps(dict(b=PROVENANCE, z=1, a=2)))
    (run/"not-python.json").write_bytes(b'{"x": 1.50, "y": [1, 2]}\n')
    (run/"duplicate-key.json").write_bytes(b'{"x": 1, "x": 2}')
    (run/"has-ref-key.json").write_bytes(b'{"$pack_ref": "00"}')
    (run/"broken.json").write_bytes(b'{"x": ')
    (run/"empty.json").write_bytes(b"")
    (run/"surrogate.json").write_text(json.dumps(dict(s="\ud800", p=PROVENANCE)))   # parses, but not UTF-8 encodable
    (run/"log.txt").write_text("训练 step 1\n"*1000, encoding="utf-8")
    (run/"binary.bin").write_bytes(bytes(range(256))*50)
    (run/"weights.pt").write_bytes(b"\0"*1000)
    (run/"half.json.partial").write_bytes(b"{")
    return base/"results"


def _originals(root: Path, base: Path):
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and p.suffix not in (".pt", ".partial")}


@pytest.fixture
def packed(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    root = _tree(tmp_path)
    out = tmp_path/"out"/"r.pack.sqlite"
    out.parent.mkdir()
    info = tool.pack(["results"], out, exclude=list(tool.DEFAULT_EXCLUDES), workers=1, log=lambda m: None)
    return tool, tmp_path, root, out, info


def test_every_file_is_rebuilt_byte_for_byte_and_weights_are_left_out(packed):
    tool, base, root, out, info = packed
    originals = _originals(root, base)
    pack = tool.Pack(out)
    stored = {p for p, _, _, _ in pack.paths()}
    assert stored == set(originals)
    assert not any(p.endswith((".pt", ".partial")) for p in stored)
    for path, data in originals.items():
        assert pack.rebuild(path) == data, path
    kinds = {p: k for p, _, _, k in pack.paths()}
    for raw in ("not-python.json", "duplicate-key.json", "has-ref-key.json", "broken.json", "empty.json", "surrogate.json", "log.txt", "binary.bin"):
        assert kinds["results/scene/job/"+raw] == "raw", raw
    for js in ("curves.json", "summary.json", "compact.json", "spaced.json", "updates/0001.json", "rows/00003.json"):
        assert kinds["results/scene/job/"+js] == "json", js
    assert tool.verify_pack(out, log=lambda m: None) == len(originals) == info["files"]


def _stream(out, table):
    import lzma
    db = sqlite3.connect(out)
    return [lzma.decompress(d).decode() for (d,) in db.execute(f"SELECT data FROM {table} ORDER BY n")]


def test_repeated_provenance_is_stored_once(packed):
    tool, base, root, out, info = packed
    config_text = tool._COMPACT(PROVENANCE["encoder"]["config"])   # innermost big block: 40 per-layer entries
    assert len(config_text) > tool.DEFAULT_MIN_BLOB
    blocks, listing = "".join(_stream(out, "blob_chunks")), "".join(_stream(out, "file_chunks"))
    assert blocks.count(config_text) == 1 and config_text not in listing   # 40+12+12+4 copies on disk
    row = [line for line in listing.split("\n") if line.startswith('[["results/scene/job/rows/00003.json"')]
    assert len(row) == 1 and '{"$pack_ref":[' in row[0] and "编码器来源" not in row[0]
    assert out.stat().st_size < sum(len(d) for d in _originals(root, base).values())/4


def test_extract_selected_files_checks_and_never_overwrites(packed):
    tool, base, root, out, _ = packed
    target = base/"unpacked"
    n = tool.extract(out, target, include=["*/updates/*"], exclude=["*/0040.json"], log=lambda m: None)
    got = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert n == 39 and got == [f"results/scene/job/updates/{i:04d}.json" for i in range(1, 40)]
    first = target/"results/scene/job/updates/0001.json"
    source = root/"scene/job/updates/0001.json"
    assert first.read_bytes() == source.read_bytes()
    assert os.stat(first).st_mtime_ns == os.stat(source).st_mtime_ns
    assert tool.extract(out, target, include=["*/updates/*"], log=lambda m: None) == 1   # only 0040 is new
    first.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        tool.extract(out, target, include=["*/updates/0001.json"], log=lambda m: None)
    assert first.read_bytes() == b"changed"


def _rewrite(out, table, change):
    import lzma
    db = sqlite3.connect(out)
    cols = "n, lines, data" if table == "file_chunks" else "n, NULL, data"
    n, lines, data = db.execute(f"SELECT {cols} FROM {table} ORDER BY n LIMIT 1").fetchone()
    db.execute(f"UPDATE {table} SET data = ? WHERE n = ?", (lzma.compress(change(lzma.decompress(data).decode()).encode()), n))
    db.commit(); db.close()


def test_a_damaged_pack_is_refused(packed):
    tool, base, root, out, _ = packed
    _rewrite(out, "blob_chunks", lambda t: t.replace('"width":1408', '"width":1409', 1))
    with pytest.raises(ValueError, match="differs from the original"):
        tool.verify_pack(out, log=lambda m: None)
    db = sqlite3.connect(out)
    db.execute("UPDATE meta SET value = '0' WHERE key = 'complete'")
    db.commit(); db.close()
    with pytest.raises(ValueError, match="not a complete"):
        tool.Pack(out)


def test_a_file_pointing_at_another_files_content_is_caught(packed):
    """The per-file sha256 must catch a record whose body was swapped for another valid one."""
    tool, base, root, out, _ = packed
    def swap(text):
        lines = text.split("\n")
        find = lambda name: next(i for i, l in enumerate(lines) if l.startswith('[["results/scene/job/updates/'+name))
        a, b = find("0007.json"), find("0008.json")
        head_a = lines[a][:lines[a].index("],")+2]
        lines[a] = head_a+lines[b][lines[b].index("],")+2:]
        return "\n".join(lines)
    _rewrite(out, "file_chunks", swap)
    pack = tool.Pack(out)
    assert pack.rebuild("results/scene/job/updates/0008.json") == (root/"scene/job/updates/0008.json").read_bytes()
    with pytest.raises(ValueError, match="differs from the original"):
        pack.rebuild("results/scene/job/updates/0007.json")


def test_worker_processes_give_the_same_files(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    root = _tree(tmp_path)
    tool.pack(["results"], tmp_path/"a.sqlite", exclude=list(tool.DEFAULT_EXCLUDES), workers=2, log=lambda m: None)
    tool.pack(["results"], tmp_path/"b.sqlite", exclude=list(tool.DEFAULT_EXCLUDES), workers=1, log=lambda m: None)
    a, b = tool.Pack(tmp_path/"a.sqlite"), tool.Pack(tmp_path/"b.sqlite")
    assert a.paths() == b.paths()
    for path, data in _originals(root, tmp_path).items():
        assert a.rebuild(path) == data
    with pytest.raises(FileExistsError):
        tool.pack(["results"], tmp_path/"a.sqlite", exclude=[], workers=1, log=lambda m: None)


def test_large_raw_files_are_split_and_rebuilt(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, "RAW_PART", 1000)
    (tmp_path/"d").mkdir()
    data = os.urandom(5500)
    (tmp_path/"d/big.log").write_bytes(data)
    (tmp_path/"d/copy.log").write_bytes(data)
    tool.pack(["d"], tmp_path/"p.sqlite", exclude=[], workers=1, log=lambda m: None)
    db = sqlite3.connect(tmp_path/"p.sqlite")
    assert db.execute("SELECT COUNT(*) FROM raw").fetchone()[0] == 6     # stored once, in 6 parts
    pack = tool.Pack(tmp_path/"p.sqlite")
    assert pack.rebuild("d/big.log") == data == pack.rebuild("d/copy.log")


def test_paths_outside_the_work_directory_are_refused(tmp_path, monkeypatch):
    tool = _tool()
    (tmp_path/"inside").mkdir()
    (tmp_path/"outside.json").write_text("{}")
    monkeypatch.chdir(tmp_path/"inside")
    with pytest.raises(ValueError, match="inside the current directory"):
        tool.pack([str(tmp_path/"outside.json")], tmp_path/"x.sqlite", exclude=[], workers=1, log=lambda m: None)


def test_exact_format_recognises_the_repository_writers():
    tool = _tool()
    value = dict(b=[1, 2.5, None, True], a="中文", c=dict(x=-0.0, y=1e-300), d=float("inf"))
    writers = [lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2, default=str)+"\n",
               lambda v: json.dumps(v, indent=1), lambda v: json.dumps(v, indent=1, ensure_ascii=False),
               lambda v: json.dumps(v, ensure_ascii=False, indent=2)+"\n", lambda v: json.dumps(v, ensure_ascii=False),
               lambda v: json.dumps(v), lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
               lambda v: json.dumps(v, indent=0), lambda v: json.dumps(v, indent=4, sort_keys=True)]
    for write in writers:
        data = write(value).encode()
        found = tool.exact_format(data)
        assert found is not None, data[:60]
        assert tool._dump(found[0], found[1]) == data
    assert math.isinf(tool.exact_format(json.dumps(value).encode())[0]["d"])
    assert tool.exact_format(b"[1, 2]\n\n") is None and tool.exact_format(b"3") is None


def test_spawned_worker_processes_work(tmp_path, monkeypatch):
    """Windows starts workers with spawn: they must import the tool by name and reach the same result."""
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    root = _tree(tmp_path)
    tool.pack(["results"], tmp_path/"s.sqlite", exclude=list(tool.DEFAULT_EXCLUDES), workers=2, log=lambda m: None,
              start_method="spawn")
    pack = tool.Pack(tmp_path/"s.sqlite")
    for path, data in _originals(root, tmp_path).items():
        assert pack.rebuild(path) == data


def _pack_with_stored_path(tool, tmp_path, stored):
    """A valid pack whose only record carries the given stored path (as a hand-made pack could)."""
    (tmp_path/"src").mkdir(exist_ok=True)
    (tmp_path/"src/a.txt").write_bytes(b"payload")
    out = tmp_path/f"p{len(list(tmp_path.glob('p*.sqlite')))}.sqlite"
    tool.pack(["src"], out, exclude=[], workers=1, log=lambda m: None)
    _rewrite(out, "file_chunks", lambda t: t.replace('"src/a.txt"', json.dumps(stored)))
    return out


@pytest.mark.parametrize("stored", ["..\\escape.txt", "a\\..\\..\\escape.txt", "\\\\server\\share\\x.txt",
                                    "C:escape.txt", "C:/escape.txt", "d/C:x.txt", "/abs.txt", "../up.txt",
                                    "a/../../up.txt", "a//b.txt", "./a.txt", "a/\x00.txt", ""])
def test_extract_refuses_paths_that_could_leave_the_output_folder(tmp_path, monkeypatch, stored):
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    out = _pack_with_stored_path(tool, tmp_path, stored)
    with pytest.raises(ValueError, match="unsafe or non-portable"):
        tool.extract(out, tmp_path/"unpacked", log=lambda m: None)
    assert not [p for p in tmp_path.rglob("*") if p.is_file() and p.read_bytes() == b"payload" and "src" not in p.parts]


def test_pack_refuses_file_names_that_cannot_be_rebuilt_everywhere(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    (tmp_path/"d").mkdir()
    (tmp_path/"d/a\\b.json").write_text("{}")
    with pytest.raises(ValueError, match="rename it or leave it out"):
        tool.pack(["d"], tmp_path/"x.sqlite", exclude=[], workers=1, log=lambda m: None)
    assert tool.pack(["d"], tmp_path/"y.sqlite", exclude=["*\\*"], workers=1, log=lambda m: None)["files"] == 0


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason="needs POSIX symlinks")
def test_extract_does_not_follow_a_linked_folder_out_of_the_output(packed):
    tool, base, root, out, _ = packed
    target, elsewhere = base/"unpacked", base/"elsewhere"
    (target/"results/scene").mkdir(parents=True)
    elsewhere.mkdir()
    (target/"results/scene/job").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        tool.extract(out, target, include=["*/updates/0001.json"], log=lambda m: None)
    assert not any(elsewhere.rglob("*"))


def test_extract_leaves_an_existing_partial_file_alone(packed):
    tool, base, root, out, _ = packed
    folder = base/"unpacked/results/scene/job/updates"
    folder.mkdir(parents=True)
    (folder/"0001.json.partial").write_bytes(b"someone else's file")
    assert tool.extract(out, base/"unpacked", include=["*/updates/0001.json"], log=lambda m: None) == 1
    assert (folder/"0001.json.partial").read_bytes() == b"someone else's file"
    assert sorted(p.name for p in folder.iterdir()) == ["0001.json", "0001.json.partial"]
    assert (folder/"0001.json").read_bytes() == (root/"scene/job/updates/0001.json").read_bytes()
    if os.name != "nt":
        mask = os.umask(0); os.umask(mask)
        assert (folder/"0001.json").stat().st_mode & 0o777 == 0o666 & ~mask


def test_a_reference_with_a_wrong_length_is_refused_even_after_the_same_offset_was_read(packed):
    """The parsed-block cache must not let a second reference with another length skip the range check."""
    tool, base, root, out, _ = packed
    pack = tool.Pack(out)
    first = next(r for r in pack.records() if r[0].endswith("rows/00003.json"))
    offset, length = json.loads(json.dumps(first[6]).split('"$pack_ref": ', 1)[1].split("}", 1)[0])
    pack.build(first)
    with pytest.raises(ValueError, match="outside the stored blocks"):
        pack._block(offset, int(pack.meta["blob_bytes"]))
    with pytest.raises(ValueError, match="outside the stored blocks"):
        pack._block(offset, True)
    with pytest.raises(ValueError):
        pack._block(offset, length-1)


def test_very_deep_json_is_kept_raw_and_rebuilt(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.chdir(tmp_path)
    (tmp_path/"d").mkdir()
    deep = ("["*400+json.dumps("x"*1300)+"]"*400).encode()
    ok = ("["*tool.MAX_DEPTH+json.dumps("x"*1300)+"]"*tool.MAX_DEPTH).encode()
    (tmp_path/"d/deep.json").write_bytes(deep)
    (tmp_path/"d/ok.json").write_bytes(ok)
    tool.pack(["d"], tmp_path/"p.sqlite", exclude=[], workers=1, log=lambda m: None)
    pack = tool.Pack(tmp_path/"p.sqlite")
    assert {p: k for p, _, _, k in pack.paths()} == {"d/deep.json": "raw", "d/ok.json": "json"}
    assert pack.rebuild("d/deep.json") == deep and pack.rebuild("d/ok.json") == ok
    assert tool.extract(tmp_path/"p.sqlite", tmp_path/"u", include=["*/ok.json"], log=lambda m: None) == 1


def test_verify_checks_the_stored_pieces_against_the_pack_record(packed):
    tool, base, root, out, _ = packed
    db = sqlite3.connect(out)
    db.execute("INSERT INTO raw VALUES (?, 0, ?)", ("f"*64, b"unused"))
    db.commit(); db.close()
    with pytest.raises(ValueError, match="raw contents"):
        tool.verify_pack(out, log=lambda m: None)
    db = sqlite3.connect(out)
    db.execute("DELETE FROM raw WHERE sha256 = ?", ("f"*64,))
    db.execute("UPDATE meta SET value = CAST(value AS INTEGER)+1 WHERE key = 'blocks'")
    db.commit(); db.close()
    with pytest.raises(ValueError, match="blocks reached"):
        tool.verify_pack(out, log=lambda m: None)
    db = sqlite3.connect(out)
    db.execute("UPDATE meta SET value = CAST(value AS INTEGER)-1 WHERE key = 'blocks'")
    db.execute("INSERT INTO blob_chunks SELECT MAX(n)+1, data FROM blob_chunks")
    db.commit(); db.close()
    with pytest.raises(ValueError, match="block pieces differ"):
        tool.verify_pack(out, log=lambda m: None)
