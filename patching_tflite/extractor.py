from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import List, Optional


TFLITE_MAGIC = b"TFL3"


def find_tflite_models_in_bytes(data: bytes, min_size: int = 1024) -> list[bytes]:
    """
    Scans a binary buffer (from disk or memory) for embedded TFLite FlatBuffer models.
    """
    models = []
    pos = 0

    while True:
        idx = data.find(TFLITE_MAGIC, pos)
        if idx == -1:
            break

        if idx >= 4:
            start_idx = idx - 4
            root_offset = struct.unpack("<I", data[start_idx : start_idx + 4])[0]

            # Plausibility check: root offset must be non-zero and reasonable
            if 0 < root_offset < 1024 * 1024:
                # Try candidate lengths
                # In FlatBuffers, we can test parsing slices or estimate length
                # Common embedded models range from a few KB to 200MB
                candidate_data = data[start_idx:]
                best_valid = None

                # Strategy: test parsing with ai_edge_litert if available
                try:
                    import ai_edge_litert.interpreter as litert
                    # Try progressive sizes or look for next magic / end of buffer
                    next_idx = data.find(TFLITE_MAGIC, idx + 4)
                    max_len = (next_idx - 4 - start_idx) if next_idx != -1 else len(candidate_data)
                    max_len = min(max_len, 256 * 1024 * 1024)

                    # Try full slice first
                    try:
                        _ = litert.Interpreter(model_content=candidate_data[:max_len])
                        best_valid = candidate_data[:max_len]
                    except Exception:
                        pass
                except ImportError:
                    pass

                if best_valid is None:
                    # Fallback length heuristic
                    next_idx = data.find(TFLITE_MAGIC, idx + 4)
                    model_len = (next_idx - 4 - start_idx) if next_idx != -1 else min(len(candidate_data), 10 * 1024 * 1024)
                    if model_len >= min_size:
                        best_valid = candidate_data[:model_len]

                if best_valid and len(best_valid) >= min_size:
                    models.append(best_valid)

        pos = idx + 4

    return models


def extract_models_from_library(
    so_path: str | Path,
    output_dir: str | Path
) -> list[Path]:
    """
    Extract all embedded .tflite models from a shared library (.so) on disk.
    """
    path = Path(so_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(f"Library not found: {path}")

    data = path.read_bytes()
    models = find_tflite_models_in_bytes(data)

    extracted_files = []
    base_name = path.stem.replace("lib", "")
    for i, m_bytes in enumerate(models):
        out_file = out / f"{base_name}_extracted_m{i:02d}.tflite"
        out_file.write_bytes(m_bytes)
        extracted_files.append(out_file)

    return extracted_files


def extract_models_from_process_memory(
    output_dir: str | Path,
    pid: Optional[int] = None
) -> list[Path]:
    """
    Scan readable memory maps of a running process (or self) for live decrypted .tflite models.
    """
    target_pid = pid or os.getpid()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    maps_file = Path(f"/proc/{target_pid}/maps")
    mem_file = Path(f"/proc/{target_pid}/mem")

    if not maps_file.exists() or not mem_file.exists():
        return []

    extracted = []
    model_count = 0

    try:
        with mem_file.open("rb", buffering=0) as mem:
            for line in maps_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                perms = parts[1]
                if not perms.startswith("r"):
                    continue  # Only read readable segments
                if perms.startswith("r-xp"):
                    continue  # Skip pure code segments

                addrs = parts[0].split("-")
                start = int(addrs[0], 16)
                end = int(addrs[1], 16)
                seg_size = end - start

                if seg_size <= 0 or seg_size > 100 * 1024 * 1024:
                    continue  # Skip unreasonably large mappings

                try:
                    mem.seek(start)
                    buf = mem.read(seg_size)
                    found = find_tflite_models_in_bytes(buf)
                    for m in found:
                        model_count += 1
                        out_p = out / f"proc_{target_pid}_mem_m{model_count:02d}.tflite"
                        out_p.write_bytes(m)
                        extracted.append(out_p)
                except (OSError, ValueError):
                    continue
    except PermissionError:
        pass

    return extracted

