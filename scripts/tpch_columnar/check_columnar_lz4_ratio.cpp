#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <lz4.h>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);

    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Could not read file: " + path);
    }

    return buffer;
}

int main() {
    const std::vector<std::string> files = {
        "data/tpch_columnar/orderkey_sf1.bin",
        "data/tpch_columnar/partkey_sf1.bin",
        "data/tpch_columnar/suppkey_sf1.bin",
        "data/tpch_columnar/quantity_sf1.bin",
        "data/tpch_columnar/extendedprice_sf1.bin",
        "data/tpch_columnar/discount_sf1.bin",
        "data/tpch_columnar/linenumber_sf1.bin",
        "data/tpch_columnar/tax_sf1.bin",
        "data/tpch_columnar/returnflag_sf1.bin",
        "data/tpch_columnar/linestatus_sf1.bin",
        "data/tpch_columnar/shipmode_sf1.bin",
        "data/tpch_columnar/shipinstruct_sf1.bin",
        "data/tpch_columnar/shipdate_sf1.bin",
        "data/tpch_columnar/regionkey_sf1.bin"
    };

    size_t total_original = 0;
    size_t total_compressed = 0;

    std::cout << std::fixed << std::setprecision(2);

    std::cout << std::left
              << std::setw(30) << "Column"
              << std::setw(15) << "Original MB"
              << std::setw(15) << "Compressed MB"
              << std::setw(15) << "Reduction %"
              << "\n";

    std::cout << "--------------------------------------------------------------------------\n";

    for (const auto& path : files) {
        std::vector<char> input = read_file(path);

        const int input_size = static_cast<int>(input.size());
        const int max_comp_size = LZ4_compressBound(input_size);

        std::vector<char> compressed(max_comp_size);

        const int comp_size = LZ4_compress_default(
            input.data(),
            compressed.data(),
            input_size,
            max_comp_size
        );

        if (comp_size <= 0) {
            throw std::runtime_error("LZ4 compression failed for: " + path);
        }

        const double original_mb = input.size() / (1024.0 * 1024.0);
        const double compressed_mb = comp_size / (1024.0 * 1024.0);
        const double reduction =
            (1.0 - static_cast<double>(comp_size) /
                   static_cast<double>(input.size())) * 100.0;

        std::string name = path.substr(path.find_last_of("/") + 1);

        std::cout << std::left
                  << std::setw(30) << name
                  << std::setw(15) << original_mb
                  << std::setw(15) << compressed_mb
                  << std::setw(15) << reduction
                  << "\n";

        total_original += input.size();
        total_compressed += static_cast<size_t>(comp_size);
    }

    std::cout << "--------------------------------------------------------------------------\n";

    const double total_original_mb =
        total_original / (1024.0 * 1024.0);

    const double total_compressed_mb =
        total_compressed / (1024.0 * 1024.0);

    const double total_reduction =
        (1.0 - static_cast<double>(total_compressed) /
               static_cast<double>(total_original)) * 100.0;

    std::cout << std::left
              << std::setw(30) << "TOTAL"
              << std::setw(15) << total_original_mb
              << std::setw(15) << total_compressed_mb
              << std::setw(15) << total_reduction
              << "\n";

    return 0;
}
// how to run?
// g++ -O3 -std=c++17 \
//   scripts/tpch_columnar/check_columnar_lz4_ratio.cpp \
//   -llz4 \
//   -o bin/check_columnar_lz4_ratio
// ./bin/check_columnar_lz4_ratio