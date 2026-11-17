"""Lossless, de-duplicated packing of a results tree for sending back (stdlib only, Python >= 3.9).

Why: the formal runs write ~2 million small JSON files, and each of them repeats the same 14-18 KB
checkpoint/encoder provenance block; zip compresses file by file, so none of that repetition is removed.

What it does (no experiment code, number, metric or source hash is touched):
  pack     walk the given paths and write ONE SQLite file. Every JSON object or array whose compact text
           is at least --min-blob characters is stored once and referenced by its position elsewhere;
           the de-duplicated text is then compressed (xz/LZMA) in 8 MiB pieces, so what remains of the
           repetition between neighbouring files is compressed away too. Files that are not JSON, or whose
           bytes the standard json module cannot reproduce exactly, are stored as raw compressed bytes.
           At the end every file is rebuilt from the pack and checked (sha256 and size).
  verify   rebuild every file in memory and check its sha256 and size.
  list     print the stored paths (optionally filtered), or a size summary by folder.
  extract  rebuild the selected files under --output; each one is checked against its sha256 first.

Rebuilt files are byte-identical to the originals (the modification time is restored too; permissions
and empty folders are not kept). Paths are stored exactly as given, relative to the current directory,
like `zip -r`. See RESULTS_RETURN_PACK.md for the commands used for the formal runs.
"""

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
import lzma
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import sys
import time

FORMAT = "pojepa-results-pack-v1"
REF_KEY = "$pack_ref"
REF_MARK = b'"$pack_ref"'
HASH_REF = re.compile(r'\{"\$pack_ref":"([0-9a-f]{32})"\}')
DEFAULT_EXCLUDES = ("*.pt", "*.partial", "*.pack.sqlite")
DEFAULT_MIN_BLOB = 1024        # nested objects/arrays at least this long (compact characters) are stored once
ROOT_BLOB = 65536              # a whole file is stored once only if it is at least this long
CHUNK = 8*1024*1024            # uncompressed size of one compressed piece
RAW_PART = 16*1024*1024
_COMPACT = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=True).encode


def _xz(data, level):
    return lzma.compress(data, format=lzma.FORMAT_XZ, preset=level)


# ---------------------------------------------------------------- exact JSON formats

def _formats(data):
    """Candidate json.dumps settings that could have produced these bytes."""
    text = data.decode("utf-8")
    newline = "\n" if text.endswith("\n") else ""
    body = text[:-1] if newline else text
    if "\n" in body:
        second = body.split("\n", 2)[1]
        indents, seps = [len(second)-len(second.lstrip(" "))], [None]
    else:
        indents, seps = [None], [None, (",", ":")]
    if any(ord(c) > 127 for c in body):
        asciis = [False]          # ensure_ascii=True would have escaped these characters
    elif "\\u" in body:
        asciis = [True, False]
    else:
        asciis = [False]          # pure ASCII without escapes: both settings write the same bytes
    for indent in indents:
        for sep in seps:
            for ensure_ascii in asciis:
                for sort_keys in (True, False):
                    yield dict(indent=indent, separators=sep, ensure_ascii=ensure_ascii, sort_keys=sort_keys,
                               newline=newline)


def _dump(value, fmt):
    kwargs = dict(indent=fmt["indent"], ensure_ascii=fmt["ensure_ascii"], sort_keys=fmt["sort_keys"],
                  allow_nan=True)
    if fmt["separators"] is not None:
        kwargs["separators"] = tuple(fmt["separators"])
    return (json.dumps(value, **kwargs)+fmt["newline"]).encode("utf-8")


def _pairs(items):
    """Keep JSON object keys in file order; duplicate keys cannot be rebuilt, so they are refused."""
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def exact_format(data):
    """(parsed value, format) when the json module rebuilds exactly these bytes, else None."""
    if REF_MARK in data:
        return None
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(value, (dict, list)):
        return None
    for fmt in _formats(data):
        try:
            if _dump(value, fmt) == data:
                return value, fmt
        except (ValueError, UnicodeEncodeError, RecursionError):
            return None
    return None


# ---------------------------------------------------------------- compaction

def compact(value, min_blob, emit, root_blob=None):
    """Compact JSON text of value in which large containers are replaced by hash refs.

    A container whose compact text has >= min_blob characters is passed to emit(sha256 hex[:32], text)
    and replaced by {"$pack_ref": "<hex>"}; children go first, so a big repeated block nested inside a
    unique record is stored once. The outermost value (root_blob given) is replaced only if it has
    >= root_blob characters, so that small unique files do not each add a stored block.
    """
    if not isinstance(value, (dict, list)):
        return _COMPACT(value)
    text = _COMPACT(value)
    if len(text) < min_blob:
        return text
    if isinstance(value, dict):
        text = "{"+",".join(_COMPACT(k)+":"+compact(v, min_blob, emit) for k, v in value.items())+"}"
    else:
        text = "["+",".join(compact(v, min_blob, emit) for v in value)+"]"
    if root_blob is not None and len(text) < root_blob:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    emit(digest, text)
    return '{"%s":"%s"}' % (REF_KEY, digest)


# ---------------------------------------------------------------- selection

def selected(path, include=(), exclude=()):
    """fnmatch on the stored posix path ('*' also matches '/', as in `zip -x`)."""
    if include and not any(fnmatch.fnmatchcase(path, p) for p in include):
        return False
    return not any(fnmatch.fnmatchcase(path, p) for p in exclude)


def _stored_path(path):
    """Relative posix path under the current directory; refuses absolute paths and '..'."""
    rel = Path(os.path.relpath(Path(path).absolute(), Path.cwd()))
    posix = PurePosixPath(rel.as_posix())
    if rel.is_absolute() or ".." in posix.parts or str(posix) in ("", "."):
        raise ValueError(f"{path} must be inside the current directory (run pack from the work directory)")
    return str(posix)


def collect(roots, exclude):
    """Sorted (stored path, real path) of every regular file under the roots, minus excluded ones."""
    found = {}
    for root in roots:
        root = Path(root)
        if root.is_file():
            items = [root]
        elif root.is_dir():
            items = []
            for folder, dirs, files in os.walk(root):
                dirs.sort()
                items.extend(Path(folder)/name for name in sorted(files))
        else:
            raise FileNotFoundError(root)
        for item in items:
            if not item.is_file():
                continue
            stored = _stored_path(item)
            if not selected(stored, (), exclude):
                continue
            if stored in found:
                raise ValueError("path given twice: "+stored)
            found[stored] = item
    return sorted(found.items())


# ---------------------------------------------------------------- pack (workers)

_SEEN = set()   # blob hashes this worker already sent during the current pack


def _reset_seen():
    _SEEN.clear()


def _encode(task):
    """Worker: one file -> its record, the blocks it adds (each sent once per worker) and raw parts."""
    stored, real, min_blob, level = task
    data = Path(real).read_bytes()
    stat = os.stat(real)
    sha = hashlib.sha256(data).hexdigest()
    pending = {}
    found = exact_format(data) if stored.lower().endswith(".json") else None
    if found is not None:
        value, fmt = found
        try:
            body = compact(value, min_blob, pending.__setitem__, root_blob=ROOT_BLOB)
            body.encode("utf-8")   # e.g. a lone surrogate from a \ud800 escape: keep such files raw
        except (ValueError, UnicodeEncodeError, RecursionError):
            found = None
    head = [stored, sha, len(data), stat.st_mtime_ns]
    if found is None:
        parts = [_xz(data[i:i+RAW_PART], level) for i in range(0, max(len(data), 1), RAW_PART)]
        return dict(head=head+["raw", None], body=None, blobs=[], raw=parts)
    if len(_SEEN) > 3000000:   # bounded memory: forgetting only means a block is sent (and ignored) again
        _SEEN.clear()
    blobs = [(d, t) for d, t in pending.items() if d not in _SEEN]   # dict order: children before parents
    _SEEN.update(pending)
    return dict(head=head+["json", [fmt["indent"], fmt["separators"], fmt["ensure_ascii"], fmt["sort_keys"],
                                    fmt["newline"]]], body=body, blobs=blobs, raw=None)


# ---------------------------------------------------------------- pack (main process)

class _Writer:
    """Main process: block positions, the two compressed streams, and in-order SQLite inserts."""

    def __init__(self, db, level, threads):
        self.db, self.level = db, level
        self.pool = ThreadPoolExecutor(threads)   # lzma releases the GIL while compressing
        self.queue = []
        self.where = {}                           # blob hash (16 bytes) -> offset << 32 | length
        self.blob_buffer, self.blob_offset, self.blob_chunk = bytearray(), 0, 0
        self.file_buffer, self.file_chunk, self.file_buffer_size = [], 0, 0

    def _submit(self, table, number, data, lines=None):
        self.queue.append((table, number, lines, self.pool.submit(_xz, bytes(data), self.level)))
        while len(self.queue) > 8 or (self.queue and self.queue[0][3].done()):
            self._drain_one()

    def _drain_one(self):
        table, number, lines, future = self.queue.pop(0)
        if table == "blob_chunks":
            self.db.execute("INSERT INTO blob_chunks VALUES (?, ?)", (number, future.result()))
        else:
            self.db.execute("INSERT INTO file_chunks VALUES (?, ?, ?)", (number, lines, future.result()))

    def _location(self, match):
        loc = self.where[bytes.fromhex(match.group(1))]
        return '{"%s":[%d,%d]}' % (REF_KEY, loc >> 32, loc & 0xFFFFFFFF)

    def add_blob(self, digest, text):
        key = bytes.fromhex(digest)
        if key in self.where:
            return
        data = HASH_REF.sub(self._location, text).encode("utf-8")
        if len(data) > 0xFFFFFFFF:
            raise ValueError("single JSON block larger than 4 GB")
        self.where[key] = self.blob_offset << 32 | len(data)
        self.blob_offset += len(data)
        self.blob_buffer += data
        while len(self.blob_buffer) >= CHUNK:
            self._submit("blob_chunks", self.blob_chunk, self.blob_buffer[:CHUNK])
            del self.blob_buffer[:CHUNK]
            self.blob_chunk += 1

    def add_file(self, head, body):
        line = ("["+json.dumps(head, ensure_ascii=False, separators=(",", ":"))+","
                + ("null" if body is None else HASH_REF.sub(self._location, body))+"]\n").encode("utf-8")
        self.file_buffer.append(line)
        self.file_buffer_size += len(line)
        if self.file_buffer_size >= CHUNK:
            self._flush_files()

    def _flush_files(self):
        if self.file_buffer:
            self._submit("file_chunks", self.file_chunk, b"".join(self.file_buffer), len(self.file_buffer))
            self.file_chunk += 1
            self.file_buffer, self.file_buffer_size = [], 0

    def close(self):
        if self.blob_buffer:
            self._submit("blob_chunks", self.blob_chunk, self.blob_buffer)
            self.blob_chunk += 1
            self.blob_buffer = bytearray()
        self._flush_files()
        while self.queue:
            self._drain_one()
        self.pool.shutdown()


def _create(path):
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE blob_chunks (n INTEGER PRIMARY KEY, data BLOB NOT NULL);
        CREATE TABLE file_chunks (n INTEGER PRIMARY KEY, lines INTEGER NOT NULL, data BLOB NOT NULL);
        CREATE TABLE raw (sha256 TEXT NOT NULL, part INTEGER NOT NULL, data BLOB NOT NULL,
                          PRIMARY KEY (sha256, part)) WITHOUT ROWID;
    """)
    return db


def pack(roots, output, *, exclude, min_blob=DEFAULT_MIN_BLOB, workers=1, level=6, verify=True, log=print):
    """Write the pack to output (never overwrites); returns counts and sizes."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    temp = output.with_name(output.name+".partial")
    if temp.exists():
        temp.unlink()
    items = collect(roots, exclude)
    started = time.time()
    total = sum(Path(real).stat().st_size for _, real in items)
    log(f"packing {len(items)} files, {total/1e9:.2f} GB, workers={workers}, min-blob={min_blob}, level={level}")
    db = _create(temp)
    writer = _Writer(db, level, max(1, min(4, workers)))
    tasks = [(stored, str(real), min_blob, level) for stored, real in items]
    _reset_seen()
    pool = multiprocessing.Pool(workers, initializer=_reset_seen) if workers > 1 else None
    json_files = 0
    try:
        results = pool.imap(_encode, tasks, chunksize=32) if pool else map(_encode, tasks)
        for done, row in enumerate(results, 1):
            for digest, text in row["blobs"]:
                writer.add_blob(digest, text)
            if row["raw"] is not None and db.execute("SELECT 1 FROM raw WHERE sha256 = ?", (row["head"][1],)).fetchone() is None:
                db.executemany("INSERT INTO raw VALUES (?, ?, ?)", [(row["head"][1], i, p) for i, p in enumerate(row["raw"])])
            writer.add_file(row["head"], row["body"])
            json_files += row["head"][4] == "json"
            if done % 100000 == 0:
                log(f"  {done}/{len(items)} files, {time.time()-started:.0f} s")
        writer.close()
    finally:
        if pool:
            pool.close()
            pool.join()
    meta = dict(format=FORMAT, created_unix=str(int(time.time())), roots=json.dumps([str(r) for r in roots]),
                exclude=json.dumps(list(exclude)), min_blob=str(min_blob), root_blob=str(ROOT_BLOB),
                chunk=str(CHUNK), level=str(level), files=str(len(items)), json_files=str(json_files),
                original_bytes=str(total), blob_bytes=str(writer.blob_offset), blob_chunks=str(writer.blob_chunk),
                blocks=str(len(writer.where)), tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                complete="1")
    db.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    db.commit()
    db.close()
    if verify and verify_pack(temp, log=log) != len(items):
        raise ValueError("verification count differs from packed count")
    temp.replace(output)
    size = output.stat().st_size
    log(f"wrote {output}: {size/1e6:.1f} MB from {total/1e9:.2f} GB ({len(items)} files, {json_files} "
        f"de-duplicated JSON, {len(writer.where)} shared blocks), {time.time()-started:.0f} s")
    return dict(files=len(items), json_files=json_files, original_bytes=total, pack_bytes=size)


# ---------------------------------------------------------------- read side

class Pack:
    """Rebuilds files from a pack; keeps a few decompressed pieces and parsed blocks in memory."""

    def __init__(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)
        meta = dict(self.db.execute("SELECT key, value FROM meta"))
        if meta.get("format") != FORMAT or meta.get("complete") != "1":
            raise ValueError(f"{path} is not a complete {FORMAT} file")
        self.meta, self.chunk = meta, int(meta["chunk"])
        self.pieces, self.blocks = OrderedDict(), {}

    def _piece(self, number):
        data = self.pieces.get(number)
        if data is None:
            row = self.db.execute("SELECT data FROM blob_chunks WHERE n = ?", (number,)).fetchone()
            if row is None:
                raise ValueError(f"missing compressed piece {number}")
            data = lzma.decompress(row[0])
            self.pieces[number] = data
            if len(self.pieces) > 8:
                self.pieces.popitem(last=False)
        else:
            self.pieces.move_to_end(number)
        return data

    def _block(self, offset, length):
        value = self.blocks.get(offset)
        if value is None:
            if offset < 0 or length <= 0 or offset+length > int(self.meta["blob_bytes"]):
                raise ValueError("block reference outside the stored blocks")
            first, last = offset//self.chunk, (offset+length-1)//self.chunk
            data = b"".join(self._piece(n) for n in range(first, last+1))
            start = offset-first*self.chunk
            value = self._resolve(json.loads(data[start:start+length].decode("utf-8"), object_pairs_hook=_pairs))
            if len(self.blocks) > 20000:
                self.blocks.clear()
            self.blocks[offset] = value
        return value

    def _resolve(self, value):
        if isinstance(value, dict):
            if len(value) == 1 and REF_KEY in value:
                offset, length = value[REF_KEY]
                return self._block(offset, length)
            return {k: self._resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v) for v in value]
        return value

    def records(self, include=(), exclude=()):
        """(path, sha256, size, mtime_ns, kind, fmt, body) of the selected files, in stored order."""
        count = 0
        for lines, data in self.db.execute("SELECT lines, data FROM file_chunks ORDER BY n"):
            rows = lzma.decompress(data).decode("utf-8").split("\n")
            if rows[-1] != "" or len(rows)-1 != lines:
                raise ValueError("file list piece is damaged")
            for line in rows[:-1]:
                head, body = json.loads(line, object_pairs_hook=_pairs)
                count += 1
                if selected(head[0], include, exclude):
                    yield (*head, body)
        if count != int(self.meta["files"]):
            raise ValueError("file count differs from the pack record")

    def build(self, record):
        """Exact original bytes of one record, checked against its sha256 and size."""
        path, sha, size, _, kind, fmt, body = record
        if kind == "json":
            indent, separators, ensure_ascii, sort_keys, newline = fmt
            data = _dump(self._resolve(body), dict(indent=indent, separators=separators, ensure_ascii=ensure_ascii,
                                                   sort_keys=sort_keys, newline=newline))
        else:
            parts = self.db.execute("SELECT data FROM raw WHERE sha256 = ? ORDER BY part", (sha,)).fetchall()
            data = b"".join(lzma.decompress(p[0]) for p in parts)
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("rebuilt file differs from the original: "+path)
        return data

    def paths(self, include=(), exclude=()):
        """(path, size, sha256, kind) of the selected files."""
        return [(r[0], r[2], r[1], r[4]) for r in self.records(include, exclude)]

    def rebuild(self, path):
        for record in self.records([path]):
            if record[0] == path:
                return self.build(record)
        raise KeyError(path)


def verify_pack(path, log=print):
    pack_ = Pack(path)
    count = 0
    for record in pack_.records():
        pack_.build(record)
        count += 1
        if count % 200000 == 0:
            log(f"  verified {count}")
    log(f"verified {count} files: every rebuilt file matches its sha256 and size")
    return count


def extract(path, output, include=(), exclude=(), log=print):
    pack_ = Pack(path)
    output = Path(output)
    written = skipped = 0
    for record in pack_.records(include, exclude):
        stored, sha, size, mtime = record[:4]
        parts = PurePosixPath(stored).parts
        if PurePosixPath(stored).is_absolute() or ".." in parts or ":" in parts[0]:
            raise ValueError("unsafe stored path: "+stored)
        target = output.joinpath(*parts)
        data = pack_.build(record)
        if target.exists():
            if target.is_file() and target.stat().st_size == size and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
                skipped += 1
                continue
            raise FileExistsError(f"{target} exists with different content; not overwritten")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name+".partial")
        temp.write_bytes(data)
        temp.replace(target)
        os.utime(target, ns=(mtime, mtime))
        written += 1
    log(f"extracted {written} files to {output} ({skipped} identical files already there)")
    return written


def summary(path, depth, include=(), exclude=()):
    totals = {}
    for record in Pack(path).records(include, exclude):
        key = "/".join(PurePosixPath(record[0]).parts[:depth])
        count, nbytes = totals.get(key, (0, 0))
        totals[key] = (count+1, nbytes+record[2])
    return totals


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("pack", help="pack files and folders into one de-duplicated SQLite file")
    a.add_argument("paths", nargs="+")
    a.add_argument("--output", required=True)
    a.add_argument("--exclude", action="append", default=[], help="fnmatch pattern on the stored path; repeatable")
    a.add_argument("--no-default-excludes", action="store_true", help=f"do not add {list(DEFAULT_EXCLUDES)}")
    a.add_argument("--min-blob", type=int, default=DEFAULT_MIN_BLOB)
    a.add_argument("--level", type=int, default=6, choices=range(10), help="xz preset (default 6)")
    a.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a.add_argument("--no-verify", action="store_true", help="skip the final rebuild check (not recommended)")
    for name, text in (("verify", "rebuild every file in memory and check it"),
                       ("list", "list stored files"), ("extract", "rebuild selected files under --output")):
        s = sub.add_parser(name, help=text)
        s.add_argument("pack")
        if name != "verify":
            s.add_argument("--include", action="append", default=[], help="fnmatch pattern; repeatable")
            s.add_argument("--exclude", action="append", default=[], help="fnmatch pattern; repeatable")
        if name == "list":
            s.add_argument("--summary", type=int, metavar="DEPTH", help="sizes summed by the first DEPTH path parts")
        if name == "extract":
            s.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.command == "pack":
        if args.min_blob < 16 or args.workers < 1:
            p.error("--min-blob must be >= 16 and --workers >= 1")
        excludes = list(args.exclude)+([] if args.no_default_excludes else list(DEFAULT_EXCLUDES))
        pack(args.paths, args.output, exclude=excludes, min_blob=args.min_blob, workers=args.workers,
             level=args.level, verify=not args.no_verify)
    elif args.command == "verify":
        verify_pack(args.pack)
    elif args.command == "list":
        if args.summary:
            for key, (count, nbytes) in sorted(summary(args.pack, args.summary, args.include, args.exclude).items()):
                print(f"{nbytes/1e6:12.1f} MB {count:9d} files  {key}")
        else:
            for stored, sha, size, _, kind, _, _ in Pack(args.pack).records(args.include, args.exclude):
                print(f"{size:12d} {kind:4s} {stored}")
    else:
        extract(args.pack, args.output, args.include, args.exclude)
    return 0


if __name__ == "__main__":
    sys.exit(main())
