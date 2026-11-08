"""usage: census.py CONFIG.json PROGRAM.exe FUNCTIONS_JSON

Instruction census over discovered code: mnemonic counts (with operand shapes) the translator must cover."""
import collections, json, os, sys
import capstone
sys.path.insert(0, os.path.dirname(__file__))
from pe import Image
from discover import Discover
img = Image(sys.argv[2], sys.argv[1]); fn = json.load(open(sys.argv[3]))
disc = Discover(img)
starts = sorted(int(k, 16) for k in fn)
disc.starts = set(starts)
cnt = collections.Counter(); prefixes = collections.Counter()
for f in starts:
    for a in disc.walk(f)["insns"]:
        i = disc.d.at(a)
        cnt[i.mnemonic] += 1
        if i.prefix[0] in (0xf0, 0xf2, 0xf3): prefixes[f"{i.prefix[0]:#x} {i.mnemonic}"] += 1
out = sorted(cnt.items(), key=lambda x: -x[1])
print(len(out), "mnemonics,", sum(cnt.values()), "instructions")
print(" ".join(f"{m}:{n}" for m, n in out))
print("prefixed:", dict(prefixes))
