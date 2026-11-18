#!/usr/bin/env python3
"""PostToolUse(Edit|Write) check of Coral config files (CLAUDE.md "Config files (hard)").

config.ini and configs/*.ini: ONE header comment line, then only sections, blank lines and settings, each
setting with an inline `; options / units` comment. No explanatory comment lines.
"""
import json
import re
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    path = (data.get("tool_input") or {}).get("file_path") or (data.get("tool_response") or {}).get("filePath") or ""
    if not re.search(r"/Software/(Coral|worktrees/[abc])/(config\.ini|configs/[^/]+\.ini)$", path):
        sys.exit(0)
    try:
        lines = open(path).read().splitlines()
    except OSError:
        sys.exit(0)
    bad = []
    if not lines or not lines[0].lstrip().startswith(";"):
        bad.append("line 1 must be the one header comment line (e.g. `; PRESET single_ring: e-/mu- single-ring fits`)")
    for n, l in enumerate(lines[1:], 2):
        s = l.strip()
        if not s or s.startswith("["):
            continue
        if s[0] in ";#":
            bad.append(f"line {n}: comment line (only ONE header line allowed; notes go in configs/README.md)")
        elif "=" in s and ";" not in s.split("=", 1)[1]:
            bad.append(f"line {n}: `{s.split('=')[0].strip()}` has no inline `; options / units` comment")
    if bad:
        print(json.dumps({"decision": "block", "reason": "Config format (CLAUDE.md): " + "; ".join(bad[:8])}))
    sys.exit(0)


if __name__ == "__main__":
    main()
