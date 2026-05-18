// Compresses every *.tbl file in a source directory using CPU LZ4 (frame
// format) and writes the output to <src_dir>/compressed/<name>.tbl.lz4.
//
// Output files are standard .lz4 frames, decompressable with the regular
// `lz4 -d` CLI (useful for verifying integrity outside this repo).
//
// Build:
//     g++ -O3 -std=c++17 src/tools/compress_tbl_lz4.cpp -o bin/compress_tbl_lz4 -llz4
//
// Usage:
//     ./bin/compress_tbl_lz4              # defaults to ./data
//     ./bin/compress_tbl_lz4 path/to/dir  # compresses *.tbl in that dir

#include <lz4frame.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Reads an entire file into a byte buffer. Follows symlinks via fstream.
static std::vector<char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p.string());

    const std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> buf(static_cast<size_t>(n));
    if (n > 0 && !f.read(buf.data(), n)) {
        throw std::runtime_error("read failed: " + p.string());
    }
    return buf;
}

// Writes a byte buffer to disk, truncating any existing file.
static void write_file(const fs::path& p, const char* data, size_t n) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write " + p.string());
    f.write(data, static_cast<std::streamsize>(n));
}

int main(int argc, char** argv) {
    try {
        const fs::path src_dir = (argc > 1) ? fs::path(argv[1]) : fs::path("data");
        const fs::path dst_dir = src_dir / "compressed";

        if (!fs::exists(src_dir) || !fs::is_directory(src_dir)) {
            std::cerr << "Source directory not found: " << src_dir << "\n";
            return 1;
        }
        fs::create_directories(dst_dir);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::left
                  << std::setw(20) << "FILE"
                  << std::right
                  << std::setw(14) << "ORIG (B)"
                  << std::setw(14) << "COMP (B)"
                  << std::setw(10) << "SAVED %"
                  << "\n";
        std::cout << "------------------------------------------------------------\n";

        size_t files_done = 0;

        for (const auto& entry : fs::directory_iterator(src_dir)) {
            const fs::path& src = entry.path();

            // Only compress *.tbl files. fs::is_regular_file follows symlinks,
            // so the data/*.tbl symlinks pointing at tpch-dbgen are picked up.
            if (src.extension() != ".tbl") continue;
            if (!fs::is_regular_file(src)) continue;

            const std::vector<char> raw = read_file(src);

            // Worst-case compressed size for the entire input as a single frame.
            const size_t bound = LZ4F_compressFrameBound(raw.size(), nullptr);
            std::vector<char> out(bound);

            // One-shot frame compression (default LZ4F preferences).
            const size_t out_size = LZ4F_compressFrame(
                out.data(), bound,
                raw.data(), raw.size(),
                nullptr);

            if (LZ4F_isError(out_size)) {
                std::cerr << "LZ4F error on " << src.filename().string()
                          << ": " << LZ4F_getErrorName(out_size) << "\n";
                return 1;
            }

            const fs::path dst = dst_dir / (src.filename().string() + ".lz4");
            write_file(dst, out.data(), out_size);

            const double saved_pct = raw.empty()
                ? 0.0
                : (1.0 - static_cast<double>(out_size) / static_cast<double>(raw.size())) * 100.0;

            std::cout << std::left
                      << std::setw(20) << src.filename().string()
                      << std::right
                      << std::setw(14) << raw.size()
                      << std::setw(14) << out_size
                      << std::setw(10) << saved_pct
                      << "\n";

            ++files_done;
        }

        if (files_done == 0) {
            std::cout << "No *.tbl files found in " << src_dir << "\n";
        } else {
            std::cout << "\nWrote " << files_done
                      << " compressed file(s) to " << dst_dir << "\n";
        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

