"""Build README.md from README.template.md by injecting generated tables, so published numbers cannot drift from the data.

Usage: python3 -I build_readme.py <blog_kv_dir>
In the template, a line `{{F:tables/<file>.md}}` is replaced with the verbatim contents of that file (relative to <blog_kv_dir>).
A missing file is an error.
"""
import os
import sys

d = sys.argv[1]
out, n = [], 0
for line in open(os.path.join(d, "README.template.md"), encoding="utf-8"):
    s = line.strip()
    if s.startswith("{{F:") and s.endswith("}}"):
        out.append(open(os.path.join(d, s[4:-2]), encoding="utf-8").read().strip("\n"))
        n += 1
    else:
        out.append(line.rstrip("\n"))
open(os.path.join(d, "README.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("README.md written;", n, "tables injected")
