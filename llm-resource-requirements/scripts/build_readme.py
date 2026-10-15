"""Build README.md from README.template.md by injecting generated tables, so published numbers cannot drift from the data.

Usage: python3 -I build_readme.py <blog_dir>
In the template, a line `{{T:<heading prefix>}}` is replaced with the body of the generated table whose `### ` heading starts with that prefix
(tables.md, tables_extra.md, tables_latency.md). A missing or ambiguous prefix is an error.
"""
import os
import sys

d = sys.argv[1]
sections = {}
for fn in ("tables.md", "tables_extra.md", "tables_latency.md"):
    cur = None
    for line in open(os.path.join(d, fn), encoding="utf-8"):
        if line.startswith("### "):
            cur = line[4:].strip()
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(line.rstrip("\n"))
out = []
for line in open(os.path.join(d, "README.template.md"), encoding="utf-8"):
    s = line.strip()
    if s.startswith("{{T:") and s.endswith("}}"):
        key = s[4:-2]
        hits = [h for h in sections if h.startswith(key)]
        if len(hits) != 1:
            raise SystemExit(f"table prefix {key!r} matched {len(hits)} sections: {hits}")
        out.append("**" + hits[0] + "**\n\n" + "\n".join(sections[hits[0]]).strip("\n"))
    else:
        out.append(line.rstrip("\n"))
open(os.path.join(d, "README.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("README.md written;", sum(1 for l in open(os.path.join(d, "README.template.md")) if l.strip().startswith("{{T:")), "tables injected")
