#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/*
  create_memmap_sorted.cpp

  Reads a BED-like file (chrom, start, end, ID, label) that is already sorted by the 4th column (ID).
  Assumes each unique ID has exactly 100 lines. Creates a memory-mapped file of shape (num_ids, 100).

  Usage:
    ./create_memmap_sorted <input_bed> <output_mmap> <id_list.txt>

  Steps:
   1) PASS 1: 
       - Read the file line by line, parse the 4th column for ID.
       - Each time you see a new ID (since file is sorted), add it to a vector.
       - Count total lines as well.
   2) Check line_count == num_ids * 100. If mismatch, abort.
   3) Create (truncate) the memmap file to hold num_ids * 100 * sizeof(int32_t).
   4) PASS 2:
       - Re-open the input BED, read line by line again.
       - Track "current_id". When it changes, increment row_idx.
       - Parse label, write it into the correct (row_idx, col_idx) in the mapped array.
       - Keep a small array of counters for each row to track how many labels written.
   5) Verify each row got exactly 100 labels.
   6) Write ID list to <id_list.txt> so row -> ID mapping is known.

  NOTE: 
   - This uses POSIX mmap, which should work on Linux/Unix. 
   - For large files, ensure enough disk space. 
   - For 3+ billion lines, expect a long runtime unless you have high I/O bandwidth. 
   - Each pass is ~3B line reads, so total ~6B line parses.
*/

static const size_t LABELS_PER_ID = 100;

// A small helper to split a tab-separated line into tokens
// We only really need columns 3 (ID) and 4 (label), but let's parse up to 5 just in case.
std::vector<std::string> split_line(const std::string& line, char delim = '\t') {
    std::vector<std::string> tokens;
    size_t start = 0;
    while (true) {
        size_t pos = line.find(delim, start);
        if (pos == std::string::npos) {
            tokens.push_back(line.substr(start));
            break;
        } else {
            tokens.push_back(line.substr(start, pos - start));
            start = pos + 1;
        }
    }
    return tokens;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <input_bed> <output_mmap> <id_list.txt>\n";
        return 1;
    }
    std::string bed_file = argv[1];
    std::string mmap_file = argv[2];
    std::string idlist_file = argv[3];

    // --------------------- PASS 1: Find unique IDs ----------------------
    std::cerr << "PASS 1: Collecting unique IDs from " << bed_file << "\n";

    std::ifstream fin(bed_file);
    if (!fin.is_open()) {
        std::cerr << "Failed to open BED file: " << bed_file << "\n";
        return 1;
    }

    std::vector<std::string> unique_ids;
    unique_ids.reserve(1000000);  // Pre-allocate if you expect many IDs

    std::string current_id;
    std::string line;
    uint64_t line_count = 0;  // Count total lines

    while (true) {
        if (!std::getline(fin, line)) {
            break; // EOF or read error
        }
        if (line.empty()) {
            continue; // skip blank lines
        }
        auto tokens = split_line(line, '\t');
        if (tokens.size() < 5) {
            std::cerr << "Line has fewer than 5 columns. Line content: " << line << "\n";
            return 1;
        }
        std::string bed_id = tokens[3];

        // If we see a new ID (since it's sorted), add to the vector
        if (bed_id != current_id) {
            unique_ids.push_back(bed_id);
            current_id = bed_id;
        }
        line_count++;
    }
    fin.close();

    uint64_t num_ids = unique_ids.size();
    std::cerr << "Found " << num_ids << " unique IDs.\n";
    std::cerr << "Total lines (non-blank) = " << line_count << "\n";

    // Validate each ID should have exactly 1000 lines
    if (line_count != (uint64_t)LABELS_PER_ID * num_ids) {
        std::cerr << "Error: line_count (" << line_count 
                  << ") != 1000 * num_ids (" << (LABELS_PER_ID * num_ids) << ")\n"
                  << "Not every ID has exactly 1000 lines, or file is malformed.\n";
        return 1;
    }

    // Write out the ID list
    {
        std::ofstream fout_idlist(idlist_file);
        if (!fout_idlist.is_open()) {
            std::cerr << "Failed to write ID list file: " << idlist_file << "\n";
            return 1;
        }
        for (auto &id : unique_ids) {
            fout_idlist << id << "\n";
        }
        std::cerr << "Wrote ID list to " << idlist_file << "\n";
    }

    // --------------------- Create memmap file ----------------------
    // We will store int32_t labels. If your labels fit in int16_t or int8_t, you can adjust.
    off_t array_size = (off_t)num_ids * LABELS_PER_ID * (off_t)sizeof(int32_t);

    // Open file for RW, create if doesn't exist, truncate if it does
    int fd = open(mmap_file.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    // Set the file to the required size
    if (ftruncate(fd, array_size) != 0) {
        std::perror("ftruncate");
        close(fd);
        return 1;
    }

    // mmap the file
    void *map_ptr = mmap(nullptr, array_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) {
        std::perror("mmap");
        close(fd);
        return 1;
    }
    // We can close fd after mmap if we want (the mapping stays valid).
    close(fd);

    // We'll treat map_ptr as int32_t*
    int32_t *data_array = reinterpret_cast<int32_t*>(map_ptr);

    // We'll keep a small array of counters for each row (to track how many labels have been written)
    std::vector<uint16_t> counters(num_ids, 0);  
    // If you expect more than 65535 lines per ID, use uint32_t. But we only have 100 lines/ID.

    // --------------------- PASS 2: Fill memmap array ----------------------
    std::cerr << "PASS 2: Populating memmap...\n";

    fin.open(bed_file);
    if (!fin.is_open()) {
        std::cerr << "Failed to re-open BED file on second pass.\n";
        munmap(map_ptr, array_size);
        return 1;
    }

    current_id.clear();
    int64_t row_idx = -1;

    while (true) {
        if (!std::getline(fin, line)) {
            break; // end
        }
        if (line.empty()) {
            continue;
        }
        auto tokens = split_line(line, '\t');
        if (tokens.size() < 5) {
            std::cerr << "Line has fewer than 5 columns. Line content: " << line << "\n";
            munmap(map_ptr, array_size);
            return 1;
        }
        std::string bed_id = tokens[3];
        std::string label_str = tokens[4];

        // If we hit a new ID, increment row_idx
        if (bed_id != current_id) {
            row_idx++;
            current_id = bed_id;
        }

        // Convert label to int
        int32_t label_val = static_cast<int32_t>(std::stoi(label_str));

        // Insert into memmap array
        uint16_t col_idx = counters[row_idx];
        if (col_idx >= LABELS_PER_ID) {
            std::cerr << "Error: ID '" << bed_id 
                      << "' has more than 100 lines.\n";
            munmap(map_ptr, array_size);
            return 1;
        }

        // row major index = row_idx*1000 + col_idx
        uint64_t offset = (uint64_t)row_idx * LABELS_PER_ID + col_idx;
        data_array[offset] = label_val;
        counters[row_idx] += 1;
    }
    fin.close();

    // Sync changes to disk
    // (msync is optional; munmap with MS_SYNC is often enough on close)
    if (msync(map_ptr, array_size, MS_SYNC) != 0) {
        std::perror("msync");
    }

    // Validate each row had exactly 1000 lines
    for (uint64_t i = 0; i < num_ids; i++) {
        if (counters[i] != LABELS_PER_ID) {
            std::cerr << "ID at row " << i << " has " << counters[i] 
                      << " != 100 lines.\n"
                      << "(ID name is " << unique_ids[i] << ")\n";
            // Not returning error here, but you could if desired.
        }
    }

    // Unmap
    if (munmap(map_ptr, array_size) != 0) {
        std::perror("munmap");
        return 1;
    }

    std::cerr << "Done. Memmap file: " << mmap_file << "\n";
    std::cerr << "ID list file: " << idlist_file << "\n";

    return 0;
}
