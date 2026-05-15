"""Orchestrator for a single (cli, task) agent run.

Usage:
    python agent_loop/run_one.py --cli claude --task swiglu \
        --max-attempts 5 --run-id swiglu_claude_smoke
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path("/home/ianbarber/Projects/KernelBench")
TASKS_DIR = PROJECT_ROOT / "agent_loop" / "tasks"
WRAPPERS_DIR = PROJECT_ROOT / "agent_loop" / "wrappers"
SANDBOX_ROOT = PROJECT_ROOT / "agent_loop" / "sandbox"
RUNS_ROOT = PROJECT_ROOT / "agent_loop" / "runs"

WRAPPER_FOR_CLI = {
    "claude": "claude_wrap.sh",
    "codex": "codex_wrap.sh",
    "kimi": "kimi_wrap.sh",
}

# Default per-CLI wall clock budget (seconds).
DEFAULT_TIMEOUT_S = 20 * 60  # 20 minutes


class GpuSampler:
    """Background nvidia-smi sampler. Records utilization.gpu samples (0..100)."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=2,
                ).stdout.strip().splitlines()
                if out:
                    try:
                        self.samples.append(int(out[0].strip()))
                    except ValueError:
                        pass
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def mean(self) -> Optional[float]:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)


def _ensure_task_files(task: str, sandbox: Path) -> None:
    src = TASKS_DIR / task
    if not src.exists():
        raise FileNotFoundError(f"task not found: {src}")
    sandbox.mkdir(parents=True, exist_ok=True)
    for fname in ("task.md", "reference.py", "harness.py",
                  "inductor_baseline_us.json"):
        s = src / fname
        if not s.exists():
            raise FileNotFoundError(f"missing task file: {s}")
        shutil.copy2(s, sandbox / fname)
    candidate = sandbox / "candidate.py"
    if not candidate.exists():
        candidate.write_text(
            "# Replace this stub with your implementation.\n"
            "# def run(x, y): ...\n"
        )


def _build_prompt(sandbox: Path, max_attempts: int) -> str:
    task_md = (sandbox / "task.md").read_text()
    footer = f"""

---

# Working environment

You are running inside a sandbox directory:

    {sandbox}

This is your working directory. You have:
- `task.md` — the task description (already shown above).
- `reference.py` — eager PyTorch reference. Do not modify.
- `harness.py` — the evaluator. Do not modify. Run with `python harness.py`.
- `candidate.py` — your work goes here. Replace its contents.
- `inductor_baseline_us.json` — baseline reference number.

You have access to Bash and file-editing tools. The Python environment is the
project's `.venv` (already activated by the wrapper) with PyTorch 2.12 nightly
cu128, Triton 3.7, and a working CUDA install (GB10 / sm_121).

# Procedure

You may iterate up to {max_attempts} times. A typical iteration is:

  1. Edit `candidate.py`.
  2. Run: `python harness.py`
  3. Read the JSON output. If `verdict` is `FAIL_MUTATION`,
     `FAIL_NONDETERMINISTIC`, or `FAIL_CORRECTNESS`, look at the relevant
     reasons fields and fix. If `verdict` is `PASS` (standard-but-not-strict)
     check `correctness_strict.reasons` and aim for `PASS_STRICT`. If
     `verdict` is `PASS_STRICT`, compare `candidate_us` to
     `inductor_baseline_us` and try to get it lower.
  4. Repeat.

When you're satisfied (or out of attempts), stop. Whatever is in
`candidate.py` at that point is your final answer.

# Final reporting

After your last `python harness.py` run, briefly state:
  - What approach you took (Triton kernel? raw CUDA? something else?).
  - Final `candidate_us` and `speedup_vs_inductor`.
  - Any caveats.

Don't spend effort on prose — focus your tokens on the kernel.
"""
    return task_md + footer


def _parse_tokens_from_log(log_text: str, cli: str) -> Optional[dict]:
    """Best-effort token-usage scraping. Returns None if nothing found."""
    if not log_text:
        return None
    # claude (text format) prints a usage line at the end like:
    #   "Total tokens: X (input: A, output: B)" or similar — varies.
    # We try a handful of regexes.
    patterns = [
        r"(?:total\s+tokens?|tokens?\s+used)[^\d]*(\d[\d,]*)",
        r"input[^\d]*(\d[\d,]*)\s+tokens?",
        r"output[^\d]*(\d[\d,]*)\s+tokens?",
        r"(\d[\d,]*)\s+input\s+tokens?",
        r"(\d[\d,]*)\s+output\s+tokens?",
    ]
    found: dict = {}
    lowered = log_text.lower()
    m = re.search(r"(\d[\d,]*)\s+input\s+tokens?", lowered)
    if m:
        found["input"] = int(m.group(1).replace(",", ""))
    m = re.search(r"(\d[\d,]*)\s+output\s+tokens?", lowered)
    if m:
        found["output"] = int(m.group(1).replace(",", ""))
    m = re.search(r"total\s+tokens?[^\d]*(\d[\d,]*)", lowered)
    if m:
        found["total"] = int(m.group(1).replace(",", ""))
    return found or None


def _run_harness(sandbox: Path) -> dict:
    """Run the harness in the sandbox; return parsed JSON (or error)."""
    py = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    proc = subprocess.run(
        [py, "harness.py"],
        cwd=str(sandbox),
        capture_output=True,
        text=True,
        timeout=300,
    )
    raw_stdout = proc.stdout
    raw_stderr = proc.stderr
    parsed: Optional[dict] = None
    # Find the last JSON object in stdout.
    if raw_stdout:
        # Strategy: find the last balanced { ... } block.
        depth = 0
        start = None
        last_obj = None
        for i, ch in enumerate(raw_stdout):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    last_obj = raw_stdout[start:i + 1]
        if last_obj is not None:
            try:
                parsed = json.loads(last_obj)
            except json.JSONDecodeError:
                parsed = None
    return {
        "exit_code": proc.returncode,
        "parsed": parsed,
        "stdout": raw_stdout,
        "stderr": raw_stderr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, choices=list(WRAPPER_FOR_CLI))
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    sandbox = SANDBOX_ROOT / args.run_id
    run_dir = RUNS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _ensure_task_files(args.task, sandbox)

    prompt_text = _build_prompt(sandbox, args.max_attempts)
    prompt_file = run_dir / "prompt.txt"
    prompt_file.write_text(prompt_text)

    wrapper = WRAPPERS_DIR / WRAPPER_FOR_CLI[args.cli]
    if not wrapper.exists():
        print(f"wrapper missing: {wrapper}", file=sys.stderr)
        return 1

    cli_log = run_dir / "cli.log"

    sampler = GpuSampler(interval_s=1.0)
    sampler.start()

    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.monotonic()

    print(f"[run_one] cli={args.cli} task={args.task} run_id={args.run_id}")
    print(f"[run_one] sandbox: {sandbox}")
    print(f"[run_one] log: {cli_log}")
    print(f"[run_one] timeout: {args.timeout_s}s")
    sys.stdout.flush()

    cli_exit_code: Optional[int] = None
    timed_out = False
    try:
        proc = subprocess.Popen(
            [
                str(wrapper),
                "--prompt-file", str(prompt_file),
                "--cwd", str(sandbox),
                "--log-file", str(cli_log),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            stdout_text, _ = proc.communicate(timeout=args.timeout_s)
            cli_exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"[run_one] TIMEOUT after {args.timeout_s}s — killing", file=sys.stderr)
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            cli_exit_code = -1
            stdout_text = ""
        # Persist captured-from-popen stdout too (wrapper already tees).
        if stdout_text:
            (run_dir / "popen_stdout.txt").write_text(stdout_text)
    finally:
        sampler.stop()

    t1 = time.monotonic()
    ended_at = dt.datetime.now(dt.timezone.utc)
    wall_clock_s = t1 - t0

    # Run the harness ourselves (don't trust agent's claims).
    print("[run_one] running harness for final evaluation...")
    sys.stdout.flush()
    harness_result_dict = None
    harness_exit_code = None
    try:
        h = _run_harness(sandbox)
        harness_exit_code = h["exit_code"]
        harness_result_dict = h["parsed"]
        (run_dir / "harness_stdout.txt").write_text(h["stdout"])
        (run_dir / "harness_stderr.txt").write_text(h["stderr"])
    except Exception as e:
        print(f"[run_one] harness invocation error: {e}", file=sys.stderr)

    # Best-effort token scrape.
    tokens = None
    try:
        if cli_log.exists():
            tokens = _parse_tokens_from_log(cli_log.read_text(errors="replace"), args.cli)
    except Exception:
        tokens = None

    result = {
        "cli": args.cli,
        "task": args.task,
        "run_id": args.run_id,
        "wall_clock_s": wall_clock_s,
        "mean_gpu_util_pct": sampler.mean,
        "gpu_util_n_samples": len(sampler.samples),
        "cli_exit_code": cli_exit_code,
        "cli_timed_out": timed_out,
        "harness_exit_code": harness_exit_code,
        "harness_result": harness_result_dict,
        "final_candidate_loc": f"agent_loop/sandbox/{args.run_id}/candidate.py",
        "sandbox_path": str(sandbox),
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "tokens": tokens,
        "max_attempts": args.max_attempts,
    }

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))

    # Short summary.
    print()
    print("=" * 60)
    print(f"run_id:           {args.run_id}")
    print(f"cli:              {args.cli}")
    print(f"wall_clock_s:     {wall_clock_s:.1f}")
    print(f"mean_gpu_util:    {sampler.mean}")
    print(f"cli_exit_code:    {cli_exit_code}{' (TIMEOUT)' if timed_out else ''}")
    print(f"harness_exit:     {harness_exit_code}")
    if harness_result_dict:
        v = harness_result_dict.get("verdict")
        cus = harness_result_dict.get("candidate_us")
        rus = harness_result_dict.get("reference_us")
        sp = harness_result_dict.get("speedup_vs_inductor")
        print(f"verdict:          {v}")
        print(f"candidate_us:     {cus}")
        print(f"reference_us:     {rus}")
        print(f"speedup_vs_ind:   {sp}")
        if not (harness_result_dict.get("correctness") or {}).get("pass"):
            print("correctness:      FAIL")
            print(f"reasons:          {(harness_result_dict.get('correctness') or {}).get('reasons')}")
    print(f"result_path:      {result_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
