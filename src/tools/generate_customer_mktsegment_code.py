#!/usr/bin/env python3
import numpy as np
from pathlib import Path

CUSTOMER_TBL = Path("data/tpch_real/sf1/customer.tbl")
REFERENCE_LOOKUP = Path("data/tpch_columnar/customer_nation_sfx40.bin")
OUTPUT_FILE = Path("data/tpch_columnar/customer_mktsegment_code_sfx40.bin")

SEGMENT_CODE = {
    "AUTOMOBILE": 0,
    "BUILDING": 1,
    "FURNITURE": 2,
    "MACHINERY": 3,
    "HOUSEHOLD": 4,
}

codes = {}

with CUSTOMER_TBL.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("|")
        custkey = int(parts[0])
        mktsegment = parts[6].strip()
        codes[custkey] = SEGMENT_CODE[mktsegment]

base_customers = max(codes.keys())

ref = np.fromfile(REFERENCE_LOOKUP, dtype=np.int32)
out = np.zeros(ref.size, dtype=np.int32)

if (ref.size - 1) % base_customers == 0:
    repeat_factor = (ref.size - 1) // base_customers
    print("Detected 1-based lookup layout")
    print("Repeat factor:", repeat_factor)

    for r in range(repeat_factor):
        offset = r * base_customers
        for custkey, code in codes.items():
            out[offset + custkey] = code

elif ref.size % base_customers == 0:
    repeat_factor = ref.size // base_customers
    print("Detected 0-based lookup layout")
    print("Repeat factor:", repeat_factor)

    for r in range(repeat_factor):
        offset = r * base_customers
        for custkey, code in codes.items():
            out[offset + custkey - 1] = code

else:
    raise RuntimeError("Could not infer customer lookup layout.")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.tofile(OUTPUT_FILE)

print("Wrote:", OUTPUT_FILE)
print("Entries:", out.size)
print("MiB:", out.nbytes / 1024 / 1024)

unique, counts = np.unique(out, return_counts=True)
print("Distribution:")
for u, c in zip(unique, counts):
    print(int(u), int(c))
