import argparse
from array import array
from pathlib import Path

import lz4.frame


DEFAULT_SCALE_FACTORS = [10]
DEFAULT_INPUT_ROOT = Path("data/tpch_real")
DEFAULT_OUTPUT_ROOT = Path("data")

LINEITEM_BUFFER_ROWS = 1_000_000


def sf_label(sf: int) -> str:
    return f"sf{sf}"


def open_text_or_lz4(path: Path):
    if str(path).endswith(".lz4"):
        return lz4.frame.open(path, mode="rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def find_table_path(input_dir: Path, table_name: str) -> Path:
    compressed_path = input_dir / "compressed" / f"{table_name}.tbl.lz4"
    raw_path = input_dir / f"{table_name}.tbl"

    if compressed_path.exists():
        return compressed_path

    if raw_path.exists():
        return raw_path

    raise FileNotFoundError(
        f"Missing both {compressed_path} and {raw_path}"
    )


def flush_buffer(buffer: array, file_handle):
    if buffer:
        buffer.tofile(file_handle)
        del buffer[:]


def convert_quantity_only(lineitem_path: Path, output_dir: Path, suffix: str) -> int:
    """
    Convert only LINEITEM.quantity.

    Outputs:
      quantity_{suffix}.bin      int32
      quantity_{suffix}_u8.bin   uint8

    This avoids memory/disk usage for unused SPJA columns.
    """
    print("\nReading LINEITEM quantity only:")
    print(f"  {lineitem_path}")

    out_quantity_i32 = output_dir / f"quantity_{suffix}.bin"
    out_quantity_u8 = output_dir / f"quantity_{suffix}_u8.bin"

    quantity_i32_buf = array("i")
    quantity_u8_buf = array("B")

    if quantity_i32_buf.itemsize != 4:
        raise RuntimeError(
            "array('i') is not 4 bytes on this platform."
        )

    row_count = 0
    min_quantity = None
    max_quantity = None

    with (
        open_text_or_lz4(lineitem_path) as f,
        open(out_quantity_i32, "wb") as f_i32,
        open(out_quantity_u8, "wb") as f_u8,
    ):
        for line in f:
            fields = line.rstrip("\n").split("|")

            if len(fields) < 5:
                continue

            quantity_value = int(float(fields[4]))

            if quantity_value < 0 or quantity_value > 255:
                raise ValueError(
                    f"quantity value {quantity_value} cannot be stored as uint8"
                )

            if min_quantity is None or quantity_value < min_quantity:
                min_quantity = quantity_value

            if max_quantity is None or quantity_value > max_quantity:
                max_quantity = quantity_value

            quantity_i32_buf.append(quantity_value)
            quantity_u8_buf.append(quantity_value)

            row_count += 1

            if row_count % LINEITEM_BUFFER_ROWS == 0:
                flush_buffer(quantity_i32_buf, f_i32)
                flush_buffer(quantity_u8_buf, f_u8)
                print(f"  processed rows: {row_count:,}")

        flush_buffer(quantity_i32_buf, f_i32)
        flush_buffer(quantity_u8_buf, f_u8)

    print(f"\nLINEITEM rows: {row_count:,}")
    print(f"quantity min: {min_quantity}")
    print(f"quantity max: {max_quantity}")

    print("\nWritten files:")
    for path in [out_quantity_i32, out_quantity_u8]:
        print(f"  {path}  ({path.stat().st_size / (1024 * 1024):.2f} MiB)")

    return row_count


def convert_scale_factor(sf: int, input_root: Path, output_root: Path):
    suffix = sf_label(sf)

    input_dir = input_root / suffix
    output_dir = output_root / f"tpch_columnar_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print(f"Converting TPC-H {suffix.upper()} quantity only")
    print("============================================================")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    lineitem_path = find_table_path(input_dir, "lineitem")

    print("\nUsing input file:")
    print(f"  LINEITEM: {lineitem_path}")

    convert_quantity_only(lineitem_path, output_dir, suffix)

    print(f"\nDone converting {suffix.upper()} quantity only.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert only TPC-H LINEITEM.quantity to int32 and uint8 binary columns."
    )

    parser.add_argument(
        "--sf",
        type=int,
        nargs="+",
        default=DEFAULT_SCALE_FACTORS,
        help="Scale factor(s) to convert, e.g. --sf 1 10",
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing sf1/sf10/etc, default: data/tpch_real",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output folder, default: data",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("TPC-H quantity-only converter")
    print("Scale factors:", args.sf)
    print("Input root:", args.input_root)
    print("Output root:", args.output_root)

    for sf in args.sf:
        convert_scale_factor(sf, args.input_root, args.output_root)

    print("\nAll requested scale factors converted successfully.")


if __name__ == "__main__":
    main()