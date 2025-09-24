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
  create_memmap_sorted.cpp (Filtered Version)

  Reads a BED-like file (chrom, start, end, ID, label) that is already
  sorted by the 4th column (ID). Creates a memory-mapped file of shape
  (num_ids, 1000) with only valid lines.

  "Valid lines" are defined as:
    - At least 5 tab-separated columns.
    - 5th column can be parsed as an integer label.

  Steps:
   1) PASS 1: 
       - Read line by line, parse ID (4th col) and label (5th col).
       - Skip any line that fails the validity check.
       - Each time we see a new (valid) ID, add it to a vector of IDs.
       - Count how many valid lines total.
   2) Create/truncate a memmap file big enough for num_ids * 1000 * int32_t.
   3) PASS 2:
       - Re-open the file, apply the same validity checks.
       - Keep a row index for each new ID. Write up to 1000 labels for that ID.
         Skip lines once that ID already has 1000.
       - Keep counters so we know how many labels we actually wrote per ID.
   4) Warn if any ID got fewer than 1000 labels.
   5) Write ID list to <id_list.txt> so row -> ID mapping is known.
*/

static const size_t LABELS_PER_ID = 1000;

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
    std::string bed_file    = argv[1];
    std::string mmap_file   = argv[2];
    std::string idlist_file = argv[3];

    // --------------------- PASS 1: Identify valid lines, collect unique IDs ----------------------
    std::cerr << "PASS 1: Collecting unique IDs (excluding invalid lines) from "
              << bed_file << "\n";

    std::ifstream fin(bed_file);
    if (!fin.is_open()) {
        std::cerr << "Failed to open BED file: " << bed_file << "\n";
        return 1;
    }

    std::vector<std::string> unique_ids;
    unique_ids.reserve(1000000);  // Pre-allocate if you expect many IDs

    std::string current_id;
    std::string line;
    uint64_t valid_line_count = 0;  // Count how many lines passed our filters

    while (true) {
        if (!std::getline(fin, line)) {
            break; // EOF or read error
        }
        if (line.empty()) {
            continue; // skip blank lines
        }
        auto tokens = split_line(line, '\t');

        // Filter #1: Must have at least 5 columns
        if (tokens.size() < 5) {
            // Skip invalid line
            continue;
        }

        // Attempt to parse label
        // If it throws (not a valid int), skip
        try {
            // We won't store it now, just test parse
            (void)std::stoi(tokens[4]);  
        } catch (...) {
            // Skip line if label not parseable
            continue;
        }

        // If we reach here, the line is valid.
        std::string bed_id = tokens[3];

        // If ID changed, add new unique ID
        if (bed_id != current_id) {
            unique_ids.push_back(bed_id);
            current_id = bed_id;
        }

        valid_line_count++;
    }
    fin.close();

    uint64_t num_ids = unique_ids.size();
    std::cerr << "Found " << num_ids << " unique IDs.\n";
    std::cerr << "Total valid lines = " << valid_line_count << "\n";

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
    // We'll store int32_t labels. If your labels fit in int16_t or int8_t, you can adjust.
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
    close(fd); // FD no longer needed after mmap

    // We'll treat map_ptr as int32_t*
    int32_t *data_array = reinterpret_cast<int32_t*>(map_ptr);

    // We'll keep a small array of counters for each row
    // (how many labels we've written for that ID so far)
    std::vector<uint16_t> counters(num_ids, 0);

    // --------------------- PASS 2: Populate the memmap ----------------------
    std::cerr << "PASS 2: Populating memmap (excluding invalid lines)...\n";

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
            break; // EOF
        }
        if (line.empty()) {
            continue;
        }

        auto tokens = split_line(line, '\t');
        // Filter #1: Must have >=5 columns
        if (tokens.size() < 5) {
            continue;
        }

        // Try parse label
        int32_t label_val;
        try {
            label_val = static_cast<int32_t>(std::stoi(tokens[4]));
        } catch (...) {
            // Skip if we can't parse label
            continue;
        }

        // Now we have a valid line
        std::string bed_id = tokens[3];

        // If ID has changed from previous line, increment row_idx
        // (We assume the file is sorted by ID, so each new ID
        //  triggers exactly one increment.)
        if (bed_id != current_id) {
            row_idx++;
            current_id = bed_id;
        }

        // If we somehow have more IDs in pass2 than in pass1,
        // row_idx might go out of range:
        if (row_idx >= (int64_t)num_ids) {
            // This should not happen if the file is truly sorted and pass1 logic matches pass2.
            std::cerr << "WARNING: Encountered more IDs than in pass1. Skipping.\n";
            break;
        }

        // If this ID already has 1000 entries, skip extras
        uint16_t col_idx = counters[row_idx];
        if (col_idx >= LABELS_PER_ID) {
            // Just skip this line
            continue;
        }

        // Write label into memmap
        uint64_t offset = (uint64_t)row_idx * LABELS_PER_ID + col_idx;
        data_array[offset] = label_val;

        // Increment count for this ID
        counters[row_idx] += 1;
    }
    fin.close();

    // Sync changes to disk (optional, but recommended)
    if (msync(map_ptr, array_size, MS_SYNC) != 0) {
        std::perror("msync");
    }

    // Warn if any ID did not end up with exactly 1000 lines
    for (uint64_t i = 0; i < num_ids; i++) {
        if (counters[i] != LABELS_PER_ID) {
            std::cerr << "WARNING: ID \"" << unique_ids[i]
                      << "\" got " << counters[i]
                      << " valid lines (expected up to 1000).\n";
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
