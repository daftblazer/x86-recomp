"""Fail if a program-specific term (tools/game_terms.txt) appears outside README.md.

usage: check_generic.py [FILES...]   (default: every tracked or new file)
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXEMPT = {"README.md", "tools/game_terms.txt"}


def main():
    terms = [l.strip().lower() for l in open(os.path.join(ROOT, "tools", "game_terms.txt"))
             if l.strip() and not l.startswith("#")]
    files = sys.argv[1:] or subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.split()
    bad = 0
    for f in files:
        rel = os.path.relpath(os.path.join(ROOT, f), ROOT)
        if rel in EXEMPT or not os.path.isfile(os.path.join(ROOT, rel)):
            continue
        try:
            text = open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace").read().lower()
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for t in terms:
                if t in line:
                    print(f"{rel}:{n}: '{t}'")
                    bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
