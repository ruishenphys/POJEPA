"""DROID 视频解码（docs/design/04-droid-prepare.md §2）。

`cv2.VideoCapture(path)` 顺序 `read()` 到返回 False，每帧 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`，得到 uint8
[720, 1280, 3]。不 seek、不读文件头帧率来取帧、不按时间取帧（文件头的宽、高、帧率只作记录）。每个进程
`cv2.setNumThreads(1)`（准备阶段 8 个进程并行，04 §5）。编码阶段（05 §2）重新解码时用同一个函数。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

FRAME_SHAPE = (720, 1280, 3)   # 04 §4 video_size


def single_thread() -> None:
    """每个进程调用一次（准备的进程池用它做 initializer）。"""
    cv2.setNumThreads(1)


def opencv_version() -> str:
    return cv2.__version__


def decode_rgb(path: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    """按顺序产出 (帧序号, RGB uint8 帧)。文件打不开时什么都不产出（帧数 0 由调用者判为检查失败）。"""
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def header(path: Path | str) -> dict:
    """文件头的宽、高、帧率（只作记录，04 §3 的 `video`）；打不开时三项都是 None。"""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"width": None, "height": None, "fps": None}
        return {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS))}
    finally:
        capture.release()


def frame_sha256(frame: np.ndarray) -> str:
    """一帧解码后 RGB 字节的 SHA-256（和金标准 `decoded_frame_sha256` 同一算法）。"""
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
