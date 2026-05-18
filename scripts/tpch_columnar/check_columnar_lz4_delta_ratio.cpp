#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <lz4.h>

static std::vector<int32_t> read_int_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::streamsize size = file.tellg();

    if (size % sizeof(int32_t) != 0) {
        throw std::runtime_error("File size is not divisible by int32: " + path);
    }

    file.seekg(0, std::ios::beg);

    std::vector<int32_t> data(size / sizeof(int32_t));

    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Could not read file: " + path);
    }

    return data;
}

static size_t lz4_compressed_size(const std::vector<int32_t>& data) {
    const int input_size = static_cast<int>(data.size() * sizeof(int32_t));
    const int max_comp_size = LZ4_compressBound(input_size);

    std::vector<char> compressed(max_comp_size);

    const int comp_size = LZ4_compress_default(
        reinterpret_cast<const char*>(data.data()),
        compressed.data(),
        input_size,
        max_comp_size
    );

    if (comp_size <= 0) {
        throw std::runtime_error("LZ4 compression failed.");
    }

    return static_cast<size_t>(comp_size);
}

static std::vector<int32_t> delta_encode(const std::vector<int32_t>& data) {
    std::vector<int32_t> delta(data.size());

    if (data.empty()) {
        return delta;
    }

    delta[0] = data[0];

    for (size_t i = 1; i < data.size(); ++i) {
        delta[i] = data[i] - data[i - 1];
    }

    return delta;
}

int main() {
    const std::vector<std::string> files = {
        "data/tpch_columnar/orderkey_sf1.bin",
        "data/tpch_columnar/partkey_sf1.bin",
        "data/tpch_columnar/suppkey_sf1.bin",
        "data/tpch_columnar/quantity_sf1.bin",
        "data/tpch_columnar/extendedprice_sf1.bin",
        "data/tpch_columnar/discount_sf1.bin",
        "data/tpch_columnar/shipdate_sf1.bin",
        "data/tpch_columnar/regionkey_sf1.bin"
    };

    size_t total_original = 0;
    size_t total_best_compressed = 0;

    std::cout << std::fixed << std::setprecision(2);

    std::cout << std::left
              << std::setw(25) << "Column"
              << std::setw(14) << "Orig MB"
              << std::setw(14) << "Raw Red%"
              << std::setw(14) << "Delta Red%"
              << std::setw(14) << "Best Red%"
              << std::setw(10) << "Chosen"
              << "\n";

    std::cout << "----------------------------------------------------------------------------------------\n";

    for (const auto& path : files) {
        std::vector<int32_t> raw = read_int_file(path);
        std::vector<int32_t> delta = delta_encode(raw);

        const size_t original_bytes = raw.size() * sizeof(int32_t);

        const size_t raw_comp = lz4_compressed_size(raw);
        const size_t delta_comp = lz4_compressed_size(delta);

        const double raw_red =
            (1.0 - static_cast<double>(raw_comp) / static_cast<double>(original_bytes)) * 100.0;

        const double delta_red =
            (1.0 - static_cast<double>(delta_comp) / static_cast<double>(original_bytes)) * 100.0;

        const bool use_delta = delta_comp < raw_comp;
        const size_t best_comp = use_delta ? delta_comp : raw_comp;
        const double best_red =
            (1.0 - static_cast<double>(best_comp) / static_cast<double>(original_bytes)) * 100.0;

        std::string name = path.substr(path.find_last_of("/") + 1);

        std::cout << std::left
                  << std::setw(25) << name
                  << std::setw(14) << (original_bytes / (1024.0 * 1024.0))
                  << std::setw(14) << raw_red
                  << std::setw(14) << delta_red
                  << std::setw(14) << best_red
                  << std::setw(10) << (use_delta ? "DELTA" : "RAW")
                  << "\n";

        total_original += original_bytes;
        total_best_compressed += best_comp;
    }

    std::cout << "----------------------------------------------------------------------------------------\n";

    const double total_red =
        (1.0 - static_cast<double>(total_best_compressed) /
               static_cast<double>(total_original)) * 100.0;

    std::cout << std::left
              << std::setw(25) << "TOTAL BEST"
              << std::setw(14) << (total_original / (1024.0 * 1024.0))
              << std::setw(14) << ""
              << std::setw(14) << ""
              << std::setw(14) << total_red
              << "\n";

    return 0;
}
// how to run?
// g++ -O3 -std=c++17 \
//   scripts/tpch_columnar/check_columnar_lz4_delta_ratio.cpp \
//   -llz4 \
//   -o bin/check_columnar_lz4_delta_ratio
// ./bin/check_columnar_lz4_delta_ratio