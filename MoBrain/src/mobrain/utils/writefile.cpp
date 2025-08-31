//
// Created by 31530 on 2025/8/10.
//

#include "writefile.hpp"

void writeBinaryFile(std::string filename, std::vector<char> data) {
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        file.seekp(0);
        file.write(data.data(),data.size());
        file.close();
    } catch (const std::exception&) {
        throw std::runtime_error("failed to write file: " + filename);
    }
}
