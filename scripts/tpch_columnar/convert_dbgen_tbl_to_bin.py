import os
import numpy as np
import lz4.frame

# Convert compressed TPC-H tables into binary columns for the benchmark.

input_dir = "data/tpch_real/sf1"
compressed_dir = os.path.join(input_dir, "compressed")

output_dir = "data/tpch_columnar"
os.makedirs(output_dir, exist_ok=True)

lineitem_path = os.path.join(compressed_dir, "lineitem.tbl.lz4")
part_path = os.path.join(compressed_dir, "part.tbl.lz4")
orders_path = os.path.join(compressed_dir, "orders.tbl.lz4")
customer_path = os.path.join(compressed_dir, "customer.tbl.lz4")


def open_text_or_lz4(path):
    # Mostly used for .tbl.lz4 files, but normal text files also work.
    if path.endswith(".lz4"):
        return lz4.frame.open(path, mode="rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


for path in [lineitem_path, part_path, orders_path, customer_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


print("Reading compressed official tpch-dbgen .tbl.lz4 files")
print(f"LINEITEM: {lineitem_path}")
print(f"PART:     {part_path}")
print(f"ORDERS:   {orders_path}")
print(f"CUSTOMER: {customer_path}")

orderkey = []
partkey = []
quantity = []
extendedprice = []
linestatus = []

# LINEITEM is the main scanned table.
with open_text_or_lz4(lineitem_path) as f:
    for line in f:
        fields = line.rstrip("\n").split("|")

        if len(fields) < 10:
            continue

        orderkey.append(int(fields[0]))
        partkey.append(int(fields[1]))
        quantity.append(int(float(fields[4])))

        # Store price as integer cents for the C++/CUDA benchmark.
        extendedprice.append(int(round(float(fields[5]) * 100.0)))

        status = fields[9]
        if status == "O":
            linestatus.append(1)
        elif status == "F":
            linestatus.append(2)
        else:
            linestatus.append(0)

orderkey = np.asarray(orderkey, dtype=np.int32)
partkey = np.asarray(partkey, dtype=np.int32)
quantity = np.asarray(quantity, dtype=np.int32)
extendedprice = np.asarray(extendedprice, dtype=np.int32)
linestatus = np.asarray(linestatus, dtype=np.int32)

print(f"LINEITEM rows: {len(partkey)}")

max_partkey = 0
part_rows = []

# PART lookup is still written for older/future query variants.
with open_text_or_lz4(part_path) as f:
    for line in f:
        fields = line.rstrip("\n").split("|")

        if len(fields) < 9:
            continue

        p_partkey = int(fields[0])
        p_type = fields[4]
        p_size = int(fields[5])

        max_partkey = max(max_partkey, p_partkey)
        part_rows.append((p_partkey, p_type, p_size))

part_category = np.zeros(max_partkey + 1, dtype=np.int32)
part_factor = np.ones(max_partkey + 1, dtype=np.int32)

for p_partkey, p_type, p_size in part_rows:
    if "BRASS" in p_type:
        category = 3
    elif "STEEL" in p_type:
        category = 2
    elif "COPPER" in p_type:
        category = 1
    else:
        category = 0

    factor = (p_size % 10) + 1

    part_category[p_partkey] = category
    part_factor[p_partkey] = factor

print(f"PART rows: {len(part_rows)}")
print(f"Max partkey: {max_partkey}")
print(f"Rows with category == 3: {np.sum(part_category == 3)}")

max_orderkey = 0
order_rows = []

# Build orderkey -> custkey lookup for the current query.
with open_text_or_lz4(orders_path) as f:
    for line in f:
        fields = line.rstrip("\n").split("|")

        if len(fields) < 2:
            continue

        o_orderkey = int(fields[0])
        o_custkey = int(fields[1])

        max_orderkey = max(max_orderkey, o_orderkey)
        order_rows.append((o_orderkey, o_custkey))

order_custkey = np.zeros(max_orderkey + 1, dtype=np.int32)

for o_orderkey, o_custkey in order_rows:
    order_custkey[o_orderkey] = o_custkey

print(f"ORDERS rows: {len(order_rows)}")
print(f"Max orderkey: {max_orderkey}")

max_custkey = 0
customer_rows = []

# Build custkey -> nation lookup.
with open_text_or_lz4(customer_path) as f:
    for line in f:
        fields = line.rstrip("\n").split("|")

        if len(fields) < 4:
            continue

        c_custkey = int(fields[0])
        c_nationkey = int(fields[3])

        max_custkey = max(max_custkey, c_custkey)
        customer_rows.append((c_custkey, c_nationkey))

customer_nation = np.zeros(max_custkey + 1, dtype=np.int32)

for c_custkey, c_nationkey in customer_rows:
    customer_nation[c_custkey] = c_nationkey

print(f"CUSTOMER rows: {len(customer_rows)}")
print(f"Max custkey: {max_custkey}")

# Write binary columns used by the benchmark.
orderkey.tofile(os.path.join(output_dir, "orderkey_sf1.bin"))
partkey.tofile(os.path.join(output_dir, "partkey_sf1.bin"))
quantity.tofile(os.path.join(output_dir, "quantity_sf1.bin"))
extendedprice.tofile(os.path.join(output_dir, "extendedprice_sf1.bin"))
linestatus.tofile(os.path.join(output_dir, "linestatus_sf1.bin"))

part_category.tofile(os.path.join(output_dir, "part_category_sf1.bin"))
part_factor.tofile(os.path.join(output_dir, "part_factor_sf1.bin"))

order_custkey.tofile(os.path.join(output_dir, "order_custkey_sf1.bin"))
customer_nation.tofile(os.path.join(output_dir, "customer_nation_sf1.bin"))

print("\nWritten binary column files:")
for name in [
    "orderkey_sf1.bin",
    "partkey_sf1.bin",
    "quantity_sf1.bin",
    "extendedprice_sf1.bin",
    "linestatus_sf1.bin",
    "part_category_sf1.bin",
    "part_factor_sf1.bin",
    "order_custkey_sf1.bin",
    "customer_nation_sf1.bin",
]:
    path = os.path.join(output_dir, name)
    print(f"  {path}  ({os.path.getsize(path) / (1024 * 1024):.2f} MiB)")

print("\nDone.")