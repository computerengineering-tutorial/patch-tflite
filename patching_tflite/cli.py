from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from patching_tflite.diff import compare_dumps
from patching_tflite.dump import load_tensor_dump
from patching_tflite.extractor import extract_models_from_library, extract_models_from_process_memory
from patching_tflite.hook_lib import find_hook_library
from patching_tflite.perfetto import export_perfetto_trace
from patching_tflite.scanner import scan_library
from patching_tflite.session import SessionPlayer


def _cmd_scan(args):
    path = Path(args.library)
    if not path.exists():
        print(f"Error: library file not found: {path}", file=sys.stderr)
        sys.exit(1)

    result = scan_library(path)
    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(result.summary())


def _cmd_extract(args):
    out_dir = Path(args.output_dir)
    if args.library:
        lib_path = Path(args.library)
        print(f"Scanning {lib_path} for embedded .tflite models...")
        extracted = extract_models_from_library(lib_path, out_dir)
    else:
        print(f"Scanning process memory for live .tflite models...")
        extracted = extract_models_from_process_memory(out_dir, pid=args.pid)

    print(f"Extracted {len(extracted)} model(s) to {out_dir}:")
    for p in extracted:
        print(f"  - {p.name} ({p.stat().st_size} bytes)")


def _cmd_diff(args):
    report = compare_dumps(args.dir_a, args.dir_b)
    report.print_report()


def _cmd_replay(args):
    session = SessionPlayer(args.session_file)
    calls = session.get_calls()
    print(f"\n=== Replaying Session: {args.session_file} ===")
    print(f"Total Calls: {len(calls)}")
    for c in calls:
        print(f"Call #{c.call_id} | Model: {c.model_name} | Subgraph: {c.subgraph_idx} | Latency: {c.elapsed_ms:.3f} ms")
        print(f"  Inputs ({len(c.inputs)}):")
        for inp in c.inputs:
            arr = SessionPlayer.decode_tensor_data(inp, as_numpy=True)
            shape = getattr(arr, "shape", len(arr))
            print(f"    - [{inp.tensor_idx}] {inp.name} shape={shape} dtype={inp.dtype}")
        print(f"  Outputs ({len(c.outputs)}):")
        for out in c.outputs:
            arr = SessionPlayer.decode_tensor_data(out, as_numpy=True)
            shape = getattr(arr, "shape", len(arr))
            print(f"    - [{out.tensor_idx}] {out.name} shape={shape} dtype={out.dtype}")
    print("=============================================\n")


def _cmd_export_trace(args):
    dst = export_perfetto_trace(args.trace_file, args.output)
    print(f"Successfully exported Perfetto trace to: {dst}")
    print("Load this file at https://ui.perfetto.dev to view the interactive timeline.")


def _cmd_run(args, unknown_args):
    if not unknown_args:
        print("Error: No command specified to run. Usage: tflite-patch run -- <command>", file=sys.stderr)
        sys.exit(1)

    try:
        hook_so = find_hook_library(args.hook_lib)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    cur_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{hook_so}:{cur_preload}" if cur_preload else str(hook_so)

    if args.trace:
        env["TFLITE_PATCH_TRACE_FILE"] = str(Path(args.trace).resolve())
    if args.tensor_limit is not None:
        env["TFLITE_PATCH_TENSOR_LIMIT"] = str(args.tensor_limit)
    if args.preview_values is not None:
        env["TFLITE_PATCH_PREVIEW_VALUES"] = str(args.preview_values)
    if args.dump_dir:
        env["TFLITE_PATCH_DUMP_DIR"] = str(Path(args.dump_dir).resolve())
    if args.dump_models:
        env["TFLITE_PATCH_DUMP_MODELS"] = str(args.dump_models)
    if args.symbol:
        env["TFLITE_PATCH_AUTO_HOOK_SYMBOL"] = str(args.symbol)
    if args.target_lib:
        env["TFLITE_PATCH_TARGET_LIB"] = str(args.target_lib)

    print(f"[tflite-patch] LD_PRELOAD={hook_so}")
    print(f"[tflite-patch] Executing: {' '.join(unknown_args)}")
    sys.exit(subprocess.call(unknown_args, env=env))


def _cmd_inspect_dump(args):
    path = Path(args.dump_file)
    if not path.exists():
        print(f"Error: dump file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.is_dir():
        files = list(path.glob("**/*.json"))
        print(f"Found {len(files)} tensor dumps in {path}:")
        for f in files:
            meta, _ = load_tensor_dump(f, as_numpy=False)
            print(f"  - {f.name}: stage={meta.get('stage')} kind={meta.get('kind')} shape={meta.get('shape')} dtype={meta.get('dtype')}")
        return

    meta, arr = load_tensor_dump(path, as_numpy=True)
    print("\n=== Tensor Dump Inspection ===")
    print(f"Stage:        {meta.get('stage')}")
    print(f"Kind:         {meta.get('kind')}")
    print(f"Model ID:     {meta.get('model_id')}")
    print(f"Model Name:   {meta.get('model_name')}")
    print(f"Tensor Name:  {meta.get('tensor_name')}")
    print(f"Dtype:        {meta.get('dtype')}")
    print(f"Shape:        {meta.get('shape')}")
    print(f"Byte size:    {meta.get('bytes')}")
    if "min" in meta and "max" in meta:
        print(f"Min / Max:    {meta.get('min'):.4f} / {meta.get('max'):.4f}")
        print(f"Mean:         {meta.get('mean'):.4f}")
    if hasattr(arr, "shape"):
        print(f"NumPy Array:  shape={arr.shape}, dtype={arr.dtype}")
        print(f"Values preview:\n{arr}")
    print("==============================\n")


def _cmd_ui(args):
    from patching_tflite.server import start_server
    start_server(host=args.host, port=args.port, demo=args.demo)


def main():
    parser = argparse.ArgumentParser(
        prog="tflite-patch",
        description="Generic TFLite Instrumentation, Hooking & Profiling Toolkit",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: scan
    p_scan = subparsers.add_parser("scan", help="Scan a shared library (.so) for TFLite symbols")
    p_scan.add_argument("library", help="Path to .so file to inspect")
    p_scan.add_argument("--json", action="store_true", help="Output in JSON format")

    # Subcommand: extract
    p_ext = subparsers.add_parser("extract", help="Extract embedded .tflite models from library or memory")
    p_ext.add_argument("--library", default=None, help="Path to .so file to scan")
    p_ext.add_argument("--pid", type=int, default=None, help="Process ID to scan memory of")
    p_ext.add_argument("--output-dir", default="./extracted_models", help="Directory to save extracted .tflite models")

    # Subcommand: diff
    p_diff = subparsers.add_parser("diff", help="Compare two dumps for numerical drift")
    p_diff.add_argument("dir_a", help="First dumps directory")
    p_diff.add_argument("dir_b", help="Second dumps directory")

    # Subcommand: replay
    p_rep = subparsers.add_parser("replay", help="Inspect/Replay a recorded .tflite_session file")
    p_rep.add_argument("session_file", help="Path to .tflite_session file")

    # Subcommand: export-trace
    p_tr = subparsers.add_parser("export-trace", help="Export trace to Perfetto / Chrome Tracing JSON")
    p_tr.add_argument("trace_file", help="Input trace log file")
    p_tr.add_argument("--output", default="trace.perfetto.json", help="Output JSON path")

    # Subcommand: run
    p_run = subparsers.add_parser("run", help="Wrap and run a command with TFLite interception")
    p_run.add_argument("--hook-lib", default=None, help="Custom path to libtflite_patch.so")
    p_run.add_argument("--target-lib", default=None, help="Target library to hook")
    p_run.add_argument("--symbol", default="TfLiteInterpreterInvoke", help="Symbol to hook")
    p_run.add_argument("--trace", default=None, help="Trace output log file")
    p_run.add_argument("--dump-dir", default=None, help="Directory to dump tensors")
    p_run.add_argument("--dump-models", default=None, help="Filter model names for dumps")
    p_run.add_argument("--tensor-limit", type=int, default=128, help="Max tensors to log")
    p_run.add_argument("--preview-values", type=int, default=4, help="Tensor value preview count")

    # Subcommand: inspect-dump
    p_dump = subparsers.add_parser("inspect-dump", help="Inspect a tensor dump JSON file or directory")
    p_dump.add_argument("dump_file", help="Path to .json dump file or dumps directory")

    # Subcommand: ui
    p_ui = subparsers.add_parser("ui", help="Launch Live Web UI Dashboard & Telemetry Server")
    p_ui.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    p_ui.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    p_ui.add_argument("--demo", action="store_true", help="Launch with real-time demo simulation")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        cmd_args = sys.argv[2:]
        if "--" in cmd_args:
            idx = cmd_args.index("--")
            parser_args = cmd_args[:idx]
            exec_args = cmd_args[idx + 1:]
        else:
            parser_args = [a for a in cmd_args if not a.startswith("-")][:1]
            exec_args = []
        parsed = p_run.parse_args(parser_args)
        _cmd_run(parsed, exec_args)
    else:
        args = parser.parse_args()
        if args.subcommand == "scan":
            _cmd_scan(args)
        elif args.subcommand == "extract":
            _cmd_extract(args)
        elif args.subcommand == "diff":
            _cmd_diff(args)
        elif args.subcommand == "replay":
            _cmd_replay(args)
        elif args.subcommand == "export-trace":
            _cmd_export_trace(args)
        elif args.subcommand == "inspect-dump":
            _cmd_inspect_dump(args)
        elif args.subcommand == "ui":
            _cmd_ui(args)


if __name__ == "__main__":
    main()
