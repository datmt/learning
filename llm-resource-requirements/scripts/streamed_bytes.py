"""Theoretical weight traffic per generated token for DENSE models, from the GGUF tensor table.

All weights are read once per token EXCEPT the token-embedding matrix, from which only one row is looked up
(unless the output head is tied to it, in which case it is read in full as the output projection).
Prints JSON lines. Usage: streamed_bytes.py <gguf-gguf-py-dir> model.gguf ...
"""
import json, sys
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
for p in sys.argv[2:]:
    r = GGUFReader(p)
    t = {x.name: x.n_bytes for x in r.tensors}
    total = sum(t.values())
    emb = t.get("token_embd.weight", 0)
    tied = "output.weight" not in t
    streamed = total if tied else total - emb
    print(json.dumps({"file": p.split("/")[-1], "tensor_bytes": total, "token_embd_bytes": emb,
                      "output_head_tied": tied, "streamed_bytes_per_token": streamed,
                      "streamed_GiB": round(streamed / 2**30, 3), "tensor_GiB": round(total / 2**30, 3)}))
