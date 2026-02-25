#!/usr/bin/env python3
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sh(cmd: str) -> str:
    p = subprocess.run(cmd, cwd=ROOT, shell=True, text=True, capture_output=True)
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def changed_files() -> list[str]:
    base = sh("git merge-base HEAD origin/main")
    if not base:
        base = "HEAD~1"
    out = sh(f"git diff --name-only {base}...HEAD")
    return [x for x in out.splitlines() if x.strip()]


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["codex", "claude", "deepseek"], required=True)
    args = ap.parse_args()

    files = changed_files()
    print(f"mode={args.mode} changed_files={len(files)}")

    # Universal secret-safety gate
    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----"),
    ]

    for rel in files:
        p = ROOT / rel
        if not p.exists() or p.is_dir():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in secret_patterns:
            if pat.search(text):
                fail(f"Potential secret pattern detected in {rel}")

    py_files = [f for f in files if f.endswith('.py')]

    if args.mode == "codex":
        # Logic/syntax sanity for changed Python files
        for rel in py_files:
            p = ROOT / rel
            try:
                compile(p.read_text(encoding='utf-8', errors='ignore'), rel, 'exec')
            except Exception as e:
                fail(f"Python syntax issue in {rel}: {e}")
        print("[PASS] codex-review passed")

    elif args.mode == "claude":
        # Architecture hygiene gate: discourage giant files without tests/docs touch
        big_changes = []
        for rel in files:
            p = ROOT / rel
            if p.exists() and p.is_file():
                try:
                    lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
                    if len(lines) > 1200:
                        big_changes.append(rel)
                except Exception:
                    pass
        if big_changes and not any(f.startswith('docs/') for f in files):
            fail(f"Large file changes detected without docs touch: {big_changes[:3]}")
        print("[PASS] claude-review passed")

    else:  # deepseek
        # Risk/security heuristics on changed code
        risky = []
        risk_tokens = [
            "eval(", "exec(", "pickle.loads(", "subprocess.Popen(", "shell=True", "os.system("
        ]
        for rel in py_files:
            p = ROOT / rel
            try:
                text = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            for tok in risk_tokens:
                if tok in text:
                    risky.append((rel, tok))
        if risky:
            fail(f"Risky patterns detected: {risky[:5]}")
        print("[PASS] deepseek-review passed")


if __name__ == "__main__":
    main()
