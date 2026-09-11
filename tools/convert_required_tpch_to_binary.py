#!/usr/bin/env python3
import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf", type=int, required=True, choices=[1, 10, 20, 30])
    parser.add_argument("--input-base", default="data/tpch_real")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"File exists but is empty: {path}")


def write_col(arr, out_dir, name, sf):
    path = os.path.join(out_dir, f"{name}_sf{sf}.bin")
    arr.astype(np.int32, copy=False).tofile(path)
    print(f"{path}: {os.path.getsize(path) / (1024 * 1024):.2f} MiB")


def main():
    args = parse_args()
    sf = args.sf

    input_dir = os.path.join(args.input_base, f"sf{sf}")
    output_dir = args.output_dir or f"data/tpch_columnar_sf{sf}"
    os.makedirs(output_dir, exist_ok=True)

    lineitem_path = os.path.join(input_dir, "lineitem.tbl")
    orders_path = os.path.join(input_dir, "orders.tbl")
    customer_path = os.path.join(input_dir, "customer.tbl")

    for p in [lineitem_path, orders_path, customer_path]:
        require_file(p)

    print(f"Converting required TPC-H tables for SF={sf}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    orderkey = []
    quantity = []
    extendedprice = []

    print("\nReading LINEITEM...")
    with open(lineitem_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 6:
                continue

            orderkey.append(int(fields[0]))
            quantity.append(int(float(fields[4])))
            extendedprice.append(int(round(float(fields[5]) * 100.0)))

    orderkey = np.asarray(orderkey, dtype=np.int32)
    quantity = np.asarray(quantity, dtype=np.int32)
    extendedprice = np.asarray(extendedprice, dtype=np.int32)

    print(f"LINEITEM rows: {len(orderkey):,}")

    max_orderkey = 0
    order_rows = []

    print("\nReading ORDERS...")
    with open(orders_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 2:
                continue

            o_orderkey = int(fields[0])
            o_custkey = int(fields[1])

            max_orderkey = max(max_orderkey, o_orderkey)
            order_rows.append((o_orderkey, o_custkey))

    order_custkey = np.zeros(max_orderkey + 1, dtype=np.int32)

    for ok, ck in order_rows:
        order_custkey[ok] = ck

    print(f"ORDERS rows: {len(order_rows):,}")
    print(f"Max orderkey: {max_orderkey:,}")

    max_custkey = 0
    customer_rows = []

    print("\nReading CUSTOMER...")
    with open(customer_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 4:
                continue

            c_custkey = int(fields[0])
            c_nationkey = int(fields[3])

            max_custkey = max(max_custkey, c_custkey)
            customer_rows.append((c_custkey, c_nationkey))

    customer_nation = np.zeros(max_custkey + 1, dtype=np.int32)

    for ck, nation in customer_rows:
        customer_nation[ck] = nation

    print(f"CUSTOMER rows: {len(customer_rows):,}")
    print(f"Max custkey: {max_custkey:,}")

    print("\nWriting binary columns...")
    write_col(orderkey, output_dir, "orderkey", sf)
    write_col(quantity, output_dir, "quantity", sf)
    write_col(extendedprice, output_dir, "extendedprice", sf)
    write_col(order_custkey, output_dir, "order_custkey", sf)
    write_col(customer_nation, output_dir, "customer_nation", sf)

    print("\nDone.")


if __name__ == "__main__":
    main()
