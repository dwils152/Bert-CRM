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
#include <stdexcept>
#include <climits> // For INT8_MIN and INT8_MAX

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
    unique_ids.reserve(1000000);

    std::string current_id;
    std::string line;
    uint64_t line_count = 0;

    while (true) {
        if (!std::getline(fin, line)) {
            break;
        }
        if (line.empty()) {
            continue;
        }
        auto tokens = split_line(line, '\t');
        if (tokens.size() < 5) {
            std::cerr << "Line has fewer than 5 columns. Line content: " << line << "\n";
            return 1;
        }
        std::string bed_id = tokens[3];

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
    off_t array_size = (off_t)num_ids * LABELS_PER_ID * (off_t)sizeof(int8_t);

    int fd = open(mmap_file.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    if (ftruncate(fd, array_size) != 0) {
        std::perror("ftruncate");
        close(fd);
        return 1;
    }

    void *map_ptr = mmap(nullptr, array_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map_ptr == MAP_FAILED) {
        std::perror("mmap");
        close(fd);
        return 1;
    }
    close(fd);

    int8_t *data_array = reinterpret_cast<int8_t*>(map_ptr);
    std::vector<uint16_t> counters(num_ids, 0);

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
            break;
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

        if (bed_id != current_id) {
            row_idx++;
            current_id = bed_id;
        }

        try {
            int32_t label_val = static_cast<int32_t>(std::stoi(label_str));
            // Check if the label fits within int8_t range
            if (label_val < INT8_MIN || label_val > INT8_MAX) {
                std::cerr << "Error: Label value " << label_val 
                          << " is out of range for int8_t. Line content: " << line << "\n";
                munmap(map_ptr, array_size);
                return 1;
            }
            int8_t label_val_int8 = static_cast<int8_t>(label_val);

            uint16_t col_idx = counters[row_idx];
            if (col_idx >= LABELS_PER_ID) {
                std::cerr << "Error: ID '" << bed_id 
                          << "' has more than 1000 lines.\n";
                munmap(map_ptr, array_size);
                return 1;
            }

            uint64_t offset = (uint64_t)row_idx * LABELS_PER_ID + col_idx;
            data_array[offset] = label_val_int8;
            counters[row_idx] += 1;
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid label value '" << label_str << "' in line: " << line << "\n";
            munmap(map_ptr, array_size);
            return 1;
        }
    }
    fin.close();

    if (msync(map_ptr, array_size, MS_SYNC) != 0) {
        std::perror("msync");
    }

    for (uint64_t i = 0; i < num_ids; i++) {
        if (counters[i] != LABELS_PER_ID) {
            std::cerr << "ID at row " << i << " has " << counters[i] 
                      << " != 1000 lines.\n"
                      << "(ID name is " << unique_ids[i] << ")\n";
        }
    }

    if (munmap(map_ptr, array_size) != 0) {
        std::perror("munmap");
        return 1;
    }

    std::cerr << "Done. Memmap file: " << mmap_file << "\n";
    std::cerr << "ID list file: " << idlist_file << "\n";

    return 0;
}