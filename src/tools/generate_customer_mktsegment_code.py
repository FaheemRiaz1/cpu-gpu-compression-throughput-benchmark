#!/usr/bin/env python3

import numpy as np

from pathlib import Path


# Source TPC-H CUSTOMER table used to recover each customer's
# market-segment category.
CUSTOMER_TBL = Path(
    "data/tpch_real/sf1/customer.tbl"
)

# Existing x40 customer lookup is used only to determine the expected
# lookup size and whether the generated layout is 0-based or 1-based.
REFERENCE_LOOKUP = Path(
    "data/tpch_columnar/customer_nation_sfx40.bin"
)

# Output dictionary-coded market-segment lookup used by the
# string-predicate SPJA benchmark.
OUTPUT_FILE = Path(
    "data/tpch_columnar/customer_mktsegment_code_sfx40.bin"
)


# Encode the five TPC-H market-segment strings as compact integer codes.
SEGMENT_CODE = {
    "AUTOMOBILE": 0,
    "BUILDING": 1,
    "FURNITURE": 2,
    "MACHINERY": 3,
    "HOUSEHOLD": 4,
}


# Build a base SF1 mapping from customer key to dictionary code.
codes = {}

with CUSTOMER_TBL.open(
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        parts = line.rstrip("\n").split("|")

        custkey = int(parts[0])
        mktsegment = parts[6].strip()

        codes[custkey] = SEGMENT_CODE[mktsegment]


# The maximum SF1 customer key determines the size of one replicated block.
base_customers = max(codes.keys())


# Load the existing x40 customer lookup to reproduce its indexing layout
# and total number of entries exactly.
ref = np.fromfile(
    REFERENCE_LOOKUP,
    dtype=np.int32
)

# Initialize the output with zeros; entries are filled according to the
# detected lookup convention below.
out = np.zeros(
    ref.size,
    dtype=np.int32
)


# A size of repeat_factor * base_customers + 1 indicates a 1-based
# lookup where index 0 is intentionally unused.
if (ref.size - 1) % base_customers == 0:

    repeat_factor = (
        ref.size - 1
    ) // base_customers

    print(
        "Detected 1-based lookup layout"
    )

    print(
        "Repeat factor:",
        repeat_factor
    )

    # Replicate the SF1 market-segment mapping across all x40 customer blocks.
    for r in range(repeat_factor):

        offset = (
            r *
            base_customers
        )

        for custkey, code in codes.items():

            out[
                offset +
                custkey
            ] = code


# An exact multiple of the base customer count indicates a 0-based
# lookup with the first customer stored at index 0.
elif ref.size % base_customers == 0:

    repeat_factor = (
        ref.size //
        base_customers
    )

    print(
        "Detected 0-based lookup layout"
    )

    print(
        "Repeat factor:",
        repeat_factor
    )

    # Replicate the SF1 mapping while converting the 1-based TPC-H
    # customer keys to 0-based array positions.
    for r in range(repeat_factor):

        offset = (
            r *
            base_customers
        )

        for custkey, code in codes.items():

            out[
                offset +
                custkey -
                1
            ] = code


# Refuse to generate a lookup if the reference size does not match
# either supported indexing convention.
else:

    raise RuntimeError(
        "Could not infer customer lookup layout."
    )


# Create the output directory if needed and write the int32 lookup
# in the binary format consumed by the SPJA benchmarks.
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.tofile(
    OUTPUT_FILE
)


# Print basic output metadata for a quick generation sanity check.
print(
    "Wrote:",
    OUTPUT_FILE
)

print(
    "Entries:",
    out.size
)

print(
    "MiB:",
    out.nbytes / 1024 / 1024
)


# Report the generated dictionary-code distribution so the replicated
# lookup can be checked for obviously unexpected values.
unique, counts = np.unique(
    out,
    return_counts=True
)

print(
    "Distribution:"
)

for u, c in zip(
    unique,
    counts
):

    print(
        int(u),
        int(c)
    )


# Run from the repository root:
#
# python3 <path-to-this-script>.py
#
# Input:
# data/tpch_real/sf1/customer.tbl
# data/tpch_columnar/customer_nation_sfx40.bin
#
# Output:
# data/tpch_columnar/customer_mktsegment_code_sfx40.bin