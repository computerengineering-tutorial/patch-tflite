from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ScanResult:
    library_path: str
    is_valid_elf: bool = False
    has_c_api: bool = False
    c_api_invoke_symbol: Optional[str] = None
    c_api_symbols: list[str] = field(default_factory=list)
    has_cpp_runtime: bool = False
    cpp_invoke_symbols: list[str] = field(default_factory=list)
    all_tflite_symbols: list[str] = field(default_factory=list)
    tflite_strings_found: list[str] = field(default_factory=list)
    recommended_symbol: Optional[str] = None
    recommended_mode: str = "unknown"

    def summary(self) -> str:
        lines = [
            f"=== TFLite Library Scan: {Path(self.library_path).name} ===",
            f"Valid ELF:            {self.is_valid_elf}",
            f"Has TFLite C API:     {self.has_c_api}",
            f"C API Invoke:         {self.c_api_invoke_symbol or 'None'}",
            f"Has C++ TFLite API:   {self.has_cpp_runtime}",
            f"C++ Invoke Symbols:   {len(self.cpp_invoke_symbols)} found",
            f"Total TFLite Symbols: {len(self.all_tflite_symbols)}",
            f"Recommended Target:   {self.recommended_symbol or 'None'}",
            f"Recommended Mode:     {self.recommended_mode}",
        ]
        if self.tflite_strings_found:
            lines.append(f"Signatures in binary: {', '.join(self.tflite_strings_found[:5])}")
        lines.append("=========================================")
        return "\n".join(lines)


def scan_library(library_path: str | Path) -> ScanResult:
    """
    Inspect a shared library (.so) for TensorFlow Lite symbols, runtime signatures,
    and recommend the best hooking strategy.
    """
    path = Path(library_path).resolve()
    result = ScanResult(library_path=str(path))

    if not path.exists():
        return result

    raw_header = path.read_bytes()[:4]
    if raw_header == b"\x7fELF":
        result.is_valid_elf = True
    else:
        return result

    # 1. Use nm or readelf to inspect symbols
    symbols = []
    try:
        proc = subprocess.run(
            ["nm", "-D", "--demangle", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3:
                    sym_name = parts[-1]
                    symbols.append(sym_name)
    except FileNotFoundError:
        pass

    if not symbols:
        # Fallback to pure string extraction if nm is unavailable or symbols stripped
        data = path.read_bytes()
        extracted = re.findall(rb"[a-zA-Z0-9_]{4,}", data)
        for s in extracted:
            try:
                symbols.append(s.decode("ascii"))
            except UnicodeDecodeError:
                pass

    tflite_syms = set()
    c_api_syms = set()
    cpp_invoke_syms = set()

    for s in symbols:
        s_lower = s.lower()
        if "tflite" in s_lower or "tensorflow" in s_lower:
            tflite_syms.add(s)

        if "tfliteinterpreterinvoke" in s_lower:
            c_api_syms.add(s)
            result.c_api_invoke_symbol = s

        if "interpreter" in s_lower and "invoke" in s_lower:
            if "tflite::interpreter::invoke" in s_lower or "_zn6tflite11interpreter6invokeev" in s_lower:
                cpp_invoke_syms.add(s)

    result.all_tflite_symbols = sorted(list(tflite_syms))
    result.c_api_symbols = sorted(list(c_api_syms))
    result.cpp_invoke_symbols = sorted(list(cpp_invoke_syms))
    result.has_c_api = bool(result.c_api_invoke_symbol)
    result.has_cpp_runtime = bool(result.cpp_invoke_symbols)

    # 2. Check binary strings for signatures
    signature_patterns = [
        b"TfLiteInterpreter",
        b"AllocateTensors",
        b"tflite::Interpreter",
        b"XNNPACK",
        b"FlexDelegate",
        b"kTfLiteOk",
    ]
    with path.open("rb") as f:
        file_bytes = f.read(10 * 1024 * 1024)  # First 10MB
        for sig in signature_patterns:
            if sig in file_bytes:
                result.tflite_strings_found.append(sig.decode("ascii", errors="ignore"))

    # 3. Determine recommendations
    if result.c_api_invoke_symbol:
        result.recommended_symbol = result.c_api_invoke_symbol
        result.recommended_mode = "symbol"
    elif result.cpp_invoke_symbols:
        result.recommended_symbol = result.cpp_invoke_symbols[0]
        result.recommended_mode = "symbol"
    elif result.tflite_strings_found:
        result.recommended_symbol = None
        result.recommended_mode = "offset"  # Library has stripped internal runtime
    else:
        result.recommended_symbol = None
        result.recommended_mode = "none"

    return result

