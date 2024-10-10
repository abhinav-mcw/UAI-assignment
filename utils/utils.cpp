#include "utils.hpp"
// template <typename T>
std::vector<std::vector<std::vector<std::vector<float>>>> read_npy_file(const std::string &filename,
                                                                        std::vector<int> &ip_shape ) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    auto N = ip_shape[0];
    auto C = ip_shape[1];
    auto H = ip_shape[2];
    auto W = ip_shape[3];
    
    // Read the entire file into a buffer
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fileSize);
    file.read(buffer.data(), fileSize);

    // Check the .npy format
    if (fileSize < 10 || std::string(buffer.data(), 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid .npy file");
    }

    // Read version
    uint8_t major_version = buffer[6];
    uint8_t minor_version = buffer[7];

    // Read header length
    uint16_t header_len;
    if (major_version == 1) {
        header_len = *reinterpret_cast<uint16_t*>(&buffer[8]);
    } else if (major_version == 2) {
        header_len = *reinterpret_cast<uint32_t*>(&buffer[8]);
    } else {
        throw std::runtime_error("Unsupported .npy version");
    }

    // Read the header
    std::string header_str(&buffer[10], header_len);

    // Parse the header to get the shape and dtype
    size_t pos = header_str.find("'descr'") + 9;
    std::string dtype = header_str.substr(pos, header_str.find("'", pos) - pos);

    pos = header_str.find("'shape'") + 8;
    std::string shape_str = header_str.substr(pos, header_str.find(")", pos) - pos + 1);

    // Parse shape string
    std::vector<size_t> shape;
    size_t start = shape_str.find("(") + 1;
    size_t end = shape_str.find(")");
    std::string dims = shape_str.substr(start, end - start);
    size_t dim;
    std::istringstream shape_stream(dims);
    while (shape_stream >> dim) {
        shape.push_back(dim);
        if (shape_stream.peek() == ',')
            shape_stream.ignore();
    }

    // Calculate total size
    size_t total_size = 1;
    for (size_t s : shape) {
        total_size *= s;
    }

    // Read the data
    size_t data_start = 10 + header_len;
    size_t data_size = fileSize - data_start;

    if (data_size != total_size * sizeof(float)) {
        throw std::runtime_error("Data size mismatch");
    }

    std::vector<float> data(total_size);
    std::vector<std::vector<std::vector<std::vector<float>>>> reshaped_data(N, std::vector<std::vector<std::vector<float>>>(
                                                      C, std::vector<std::vector<float>>(
                                                          H, std::vector<float>(W, 0))));
    std::memcpy(data.data(), &buffer[data_start], data_size);

    // Fill the 4D vector with data from the 1D vector
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t index = n * C * H * W + c * H * W + h * W + w;
                    reshaped_data[n][c][h][w] = data[index];
                }
            }
        }
    }
    return reshaped_data;
}

void write_to_binary(const std::string &filename, const std::vector<float> &data)
{
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    file.close();
}