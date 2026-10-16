import sys, glob, json
from gguf import GGUFReader
def val(f):
    try:
        if f.types and f.types[0].name == "STRING": return bytes(f.parts[-1]).decode()
        d = f.parts[f.data[0]] if len(f.data)==1 else [f.parts[i].tolist() for i in f.data]
        return d.tolist()[0] if hasattr(d,"tolist") and len(f.data)==1 and d.size==1 else (d.tolist() if hasattr(d,"tolist") else d)
    except Exception as e: return f"?{e}"
out={}
for p in sys.argv[1:]:
    r=GGUFReader(p); m={}
    for k,f in r.fields.items():
        if k.startswith("tokenizer.") : continue
        v=val(f)
        if isinstance(v,list) and len(v)>64: v=f"<list {len(v)}: {v[:6]}...>"
        m[k]=v
    out[p.split('/')[-1]]=m
json.dump(out,sys.stdout,indent=1,default=str)
