#!/usr/bin/env python3

"""Manage the Coral worktree slots that let several agents work at once.

There is one primary checkout (Software/Coral, slot `main`, reserved for
interactive work) plus three agent slots:

    Software/worktrees/a   Software/worktrees/b   Software/worktrees/c

Each slot is an independent git worktree, so three agents can hold three
different branches simultaneously without `git checkout` fighting. The same
layout exists on the server, and `exp.py submit --slot <s>` runs from the
matching slot on whichever machine the run targets.

    ./wt.py list                        slots here and on the server
    ./wt.py add a ben/sandbox           create slot 'a' on this laptop
    ./wt.py add a ben/sandbox --server  create it on the cluster too
    ./wt.py add a ben/sandbox --both    both at once
    ./wt.py rm a [--server|--both]      remove a slot
    ./wt.py prune [--both]              drop stale worktree registrations

No conda environment is needed per slot: Coral's editable install is a plain
path entry in site-packages, and PYTHONPATH takes precedence over it, so
pointing PYTHONPATH at a slot's src/ is enough to run that slot's code. The
runners and submit scripts do this for you via --coral-root.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

SCRIPTS_DIR  = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPTS_DIR.parents[1]
PRIMARY      = RESEARCH_DIR / "Software/Coral"

REMOTE_HOST = "cca.in2p3.fr"
REMOTE_BASE = "/sps/hyperk/bcarew"
REMOTE_PRIMARY = f"{REMOTE_BASE}/Software/Coral"

SLOTS = ("a", "b", "c")

SSH_OPTS = ["-o", "ControlMaster=auto",
            "-o", "ControlPath=/tmp/.exp-ssh-%r@%h:%p",
            "-o", "ControlPersist=600"]


def local_slot_path(slot: str) -> Path:
    return RESEARCH_DIR / "Software/worktrees" / slot


def remote_slot_path(slot: str) -> str:
    return f"{REMOTE_BASE}/Software/worktrees/{slot}"


def run(cmd: list, capture=False):
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd]).returncode


def ssh(command: str, capture=False):
    cmd = ["ssh", *SSH_OPTS, REMOTE_HOST, command]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    print(f"+ ssh {REMOTE_HOST} {command!r}")
    return subprocess.run(cmd).returncode


# -----------------------------
# list
# -----------------------------

def _local_slots() -> list:
    rows = []
    for label, path in [("main", PRIMARY)] + [(s, local_slot_path(s)) for s in SLOTS]:
        if not (path / "src").is_dir() and not path.exists():
            rows.append((label, str(path), "-", "-", "absent"))
            continue
        rc, branch, _ = run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture=True)
        _, sha, _ = run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture=True)
        _, dirt, _ = run(["git", "-C", str(path), "status", "--porcelain"], capture=True)
        rows.append((label, str(path), branch if rc == 0 else "?", sha,
                     "dirty" if dirt else "clean"))
    return rows


def _remote_slots() -> list:
    script = "; ".join(
        f'echo "{label}|{path}|$(git -C {path} rev-parse --abbrev-ref HEAD 2>/dev/null)'
        f'|$(git -C {path} rev-parse --short HEAD 2>/dev/null)'
        f'|$(test -n "$(git -C {path} status --porcelain 2>/dev/null)" && echo dirty || echo clean)"'
        for label, path in [("main", REMOTE_PRIMARY)] + [(s, remote_slot_path(s)) for s in SLOTS])
    rc, out, err = ssh(script, capture=True)
    if rc != 0:
        return [("(server unreachable)", err, "", "", "")]
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 5:
            label, path, branch, sha, state = parts
            rows.append((label, path, branch or "-", sha or "-",
                         state if branch else "absent"))
    return rows


def _print_rows(title: str, rows: list):
    print(title)
    print(f"  {'SLOT':<6} {'BRANCH':<32} {'SHA':<10} {'STATE':<8} PATH")
    for label, path, branch, sha, state in rows:
        print(f"  {label:<6} {branch:<32} {sha:<10} {state:<8} {path}")
    print()


def cmd_list(args):
    _print_rows("=== LAPTOP ===", _local_slots())
    if not args.local:
        _print_rows("=== SERVER (CC-IN2P3) ===", _remote_slots())
    print("Use a slot with:  ./exp.py submit --slot <a|b|c> ...")


# -----------------------------
# add / rm / prune
# -----------------------------

def cmd_add(args):
    slot, branch = args.slot, args.branch
    if slot not in SLOTS:
        raise SystemExit(f"slot must be one of {SLOTS} (got '{slot}')")

    if args.where in ("local", "both"):
        path = local_slot_path(slot)
        path.parent.mkdir(parents=True, exist_ok=True)
        if (path / ".git").exists() or path.is_dir() and any(path.iterdir()):
            print(f"[LOCAL] slot '{slot}' already exists at {path} — checking out {branch}")
            rc = run(["git", "-C", str(path), "checkout", branch])
        else:
            # -B so an existing branch is reused and a new one is created from
            # the current HEAD; a worktree cannot share a branch with another
            # worktree, which is exactly the collision we want git to enforce.
            rc = run(["git", "-C", str(PRIMARY), "worktree", "add", str(path), branch])
            if rc != 0:
                print(f"[LOCAL] '{branch}' is not an existing branch — creating it")
                rc = run(["git", "-C", str(PRIMARY), "worktree", "add", "-b", branch, str(path)])
        if rc != 0:
            raise SystemExit(f"[LOCAL] failed to create slot '{slot}'")
        print(f"[LOCAL] slot '{slot}' -> {path} @ {branch}\n")

    if args.where in ("server", "both"):
        path = remote_slot_path(slot)
        rc = ssh(f"mkdir -p {REMOTE_BASE}/Software/worktrees && "
                 f"(git -C {REMOTE_PRIMARY} worktree add {path} {branch} || "
                 f" git -C {REMOTE_PRIMARY} fetch origin {branch} && "
                 f" git -C {REMOTE_PRIMARY} worktree add {path} {branch})")
        if rc != 0:
            raise SystemExit(f"[SERVER] failed to create slot '{slot}'")
        print(f"[SERVER] slot '{slot}' -> {path} @ {branch}\n")


def cmd_rm(args):
    slot = args.slot
    if slot not in SLOTS:
        raise SystemExit(f"slot must be one of {SLOTS} (got '{slot}')")
    if args.where in ("local", "both"):
        run(["git", "-C", str(PRIMARY), "worktree", "remove", str(local_slot_path(slot)),
             *(["--force"] if args.force else [])])
    if args.where in ("server", "both"):
        ssh(f"git -C {REMOTE_PRIMARY} worktree remove {remote_slot_path(slot)}"
            + (" --force" if args.force else ""))


def cmd_prune(args):
    if args.where in ("local", "both"):
        run(["git", "-C", str(PRIMARY), "worktree", "prune", "-v"])
    if args.where in ("server", "both"):
        ssh(f"git -C {REMOTE_PRIMARY} worktree prune -v")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show slots here and on the server")
    p_list.add_argument("--local", action="store_true", help="skip the server query")
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="create a slot")
    p_add.add_argument("slot", choices=SLOTS)
    p_add.add_argument("branch")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("rm", help="remove a slot")
    p_rm.add_argument("slot", choices=SLOTS)
    p_rm.add_argument("--force", action="store_true", help="remove even if dirty")
    p_rm.set_defaults(func=cmd_rm)

    p_prune = sub.add_parser("prune", help="drop stale worktree registrations")
    p_prune.set_defaults(func=cmd_prune)

    for p in (p_add, p_rm, p_prune):
        where = p.add_mutually_exclusive_group()
        where.add_argument("--server", dest="where", action="store_const", const="server",
                           help="act on the cluster only")
        where.add_argument("--both", dest="where", action="store_const", const="both",
                           help="act on the laptop and the cluster")
        p.set_defaults(where="local")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
