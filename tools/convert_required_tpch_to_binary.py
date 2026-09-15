#!/usr/bin/env python3

import argparse
import os

import numpy as np


# Parse the requested TPC-H scale factor and optional input/output locations.
# Only the scale factors used by this preprocessing workflow are accepted.
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sf",
        type=int,
        required=True,
        choices=[1, 10, 20, 30]
    )

    parser.add_argument(
        "--input-base",
        default="data/tpch_real"
    )

    parser.add_argument(
        "--output-dir",
        default=None
    )

    return parser.parse_args()


# Validate required TPC-H source files before starting conversion.
# Empty files are rejected to avoid silently producing incomplete columns.
def require_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    if os.path.getsize(path) == 0:
        raise RuntimeError(
            f"File exists but is empty: {path}"
        )


# Write one int32 column in the binary format consumed by the benchmarks.
# The scale factor is included in the filename to keep datasets separate.
def write_col(arr, out_dir, name, sf):

    path = os.path.join(
        out_dir,
        f"{name}_sf{sf}.bin"
    )

    arr.astype(
        np.int32,
        copy=False
    ).tofile(path)

    print(
        f"{path}: "
        f"{os.path.getsize(path) / (1024 * 1024):.2f} MiB"
    )


def main():

    args = parse_args()

    sf = args.sf


    # Resolve the TPC-H dbgen directory for the requested scale factor.
    input_dir = os.path.join(
        args.input_base,
        f"sf{sf}"
    )

    # Use the default scale-specific columnar directory unless the
    # caller explicitly supplies another output location.
    output_dir = (
        args.output_dir
        or f"data/tpch_columnar_sf{sf}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # Only the three TPC-H tables required by the SPJA workload are read.
    lineitem_path = os.path.join(
        input_dir,
        "lineitem.tbl"
    )

    orders_path = os.path.join(
        input_dir,
        "orders.tbl"
    )

    customer_path = os.path.join(
        input_dir,
        "customer.tbl"
    )


    # Verify all required input tables before performing any conversion.
    for p in [
        lineitem_path,
        orders_path,
        customer_path
    ]:
        require_file(p)


    print(
        f"Converting required TPC-H tables for SF={sf}"
    )

    print(
        f"Input:  {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )


    # LINEITEM provides the three fact columns used directly by
    # the analytical SPJA query.
    orderkey = []
    quantity = []
    extendedprice = []


    print(
        "\nReading LINEITEM..."
    )


    with open(
        lineitem_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            fields = line.rstrip("\n").split("|")

            # Skip malformed rows that do not contain the required columns.
            if len(fields) < 6:
                continue


            # Keep TPC-H order keys as integer identifiers.
            orderkey.append(
                int(fields[0])
            )


            # Quantity is stored as int32 for the predicate quantity > 25.
            quantity.append(
                int(
                    float(fields[4])
                )
            )


            # Preserve extendedprice exactly to cent precision by converting
            # the decimal currency value to integer cents.
            extendedprice.append(
                int(
                    round(
                        float(fields[5]) *
                        100.0
                    )
                )
            )


    # Convert the accumulated Python lists into the int32 layout expected
    # by the CPU/GPU benchmark implementations.
    orderkey = np.asarray(
        orderkey,
        dtype=np.int32
    )

    quantity = np.asarray(
        quantity,
        dtype=np.int32
    )

    extendedprice = np.asarray(
        extendedprice,
        dtype=np.int32
    )


    print(
        f"LINEITEM rows: {len(orderkey):,}"
    )


    # ORDERS is converted into a direct orderkey -> custkey lookup.
    # Index 0 remains unused because TPC-H keys are positive and the
    # benchmark accesses the lookup directly by order key.
    max_orderkey = 0
    order_rows = []


    print(
        "\nReading ORDERS..."
    )


    with open(
        orders_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            fields = line.rstrip("\n").split("|")

            if len(fields) < 2:
                continue


            o_orderkey = int(
                fields[0]
            )

            o_custkey = int(
                fields[1]
            )


            max_orderkey = max(
                max_orderkey,
                o_orderkey
            )

            order_rows.append(
                (
                    o_orderkey,
                    o_custkey
                )
            )


    # Allocate max_key + 1 entries so each TPC-H order key can be used
    # directly as an array index.
    order_custkey = np.zeros(
        max_orderkey + 1,
        dtype=np.int32
    )


    for ok, ck in order_rows:

        order_custkey[ok] = ck


    print(
        f"ORDERS rows: {len(order_rows):,}"
    )

    print(
        f"Max orderkey: {max_orderkey:,}"
    )


    # CUSTOMER is converted into a direct custkey -> nationkey lookup
    # used by the second join/filter stage of the SPJA query.
    max_custkey = 0
    customer_rows = []


    print(
        "\nReading CUSTOMER..."
    )


    with open(
        customer_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            fields = line.rstrip("\n").split("|")

            if len(fields) < 4:
                continue


            c_custkey = int(
                fields[0]
            )

            c_nationkey = int(
                fields[3]
            )


            max_custkey = max(
                max_custkey,
                c_custkey
            )

            customer_rows.append(
                (
                    c_custkey,
                    c_nationkey
                )
            )


    # As with order_custkey, allocate one extra element so the positive
    # customer key can be used directly as the lookup index.
    customer_nation = np.zeros(
        max_custkey + 1,
        dtype=np.int32
    )


    for ck, nation in customer_rows:

        customer_nation[ck] = nation


    print(
        f"CUSTOMER rows: {len(customer_rows):,}"
    )

    print(
        f"Max custkey: {max_custkey:,}"
    )


    # Write all five columns required by the SPJA benchmark.
    print(
        "\nWriting binary columns..."
    )


    write_col(
        orderkey,
        output_dir,
        "orderkey",
        sf
    )

    write_col(
        quantity,
        output_dir,
        "quantity",
        sf
    )

    write_col(
        extendedprice,
        output_dir,
        "extendedprice",
        sf
    )

    write_col(
        order_custkey,
        output_dir,
        "order_custkey",
        sf
    )

    write_col(
        customer_nation,
        output_dir,
        "customer_nation",
        sf
    )


    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()


# Run from the repository root, for example:
#
# python3 <path-to-this-script>.py --sf 10
#
# Optional:
#   --input-base <directory>
#   --output-dir <directory>
#
# Default input for --sf 10:
# data/tpch_real/sf10/
#
# Default output:
# data/tpch_columnar_sf10/
#
# Generated files:
# orderkey_sf10.bin
# quantity_sf10.bin
# extendedprice_sf10.bin
# order_custkey_sf10.bin
# customer_nation_sf10.bin