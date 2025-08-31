
#include "readfile.hpp"
#include <iostream>
std::vector<char> ateBinaryFile(const std::string& filename) {

    try
    {
        std::cout << "begin" << std::endl;


        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        const size_t fileSize = file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("failed to read file: " + filename);
    }

}
