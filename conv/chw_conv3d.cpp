#include <iostream>
#include <vector>
#include "../utils/utils.hpp"

using namespace std;

using Vector4D = vector<std::vector<std::vector<std::vector<float>>>>;

void chw_conv3d(const Vector4D &input,
                const Vector4D &kernel,
                Vector4D &output,
                int padding, int stride)
{
    int batches = input.size();
    int input_channels = input[0].size();
    int input_height = input[0][0].size();
    int input_width = input[0][0][0].size();

    int output_channels = kernel.size();
    int kernel_height = kernel[0][0].size();
    int kernel_width = kernel[0][0][0].size();

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    output.assign(batches, vector<vector<vector<float>>>(
                               output_channels, vector<vector<float>>(
                                                    output_height, vector<float>(output_width, 0))));

    // Create padded input
    vector<vector<vector<vector<float>>>> padded_input(batches, vector<vector<vector<float>>>(
                                                                    input_channels, vector<vector<float>>(
                                                                                        input_height + 2 * padding, vector<float>(input_width + 2 * padding, 0))));

    // Copy input to padded input
    for (int b = 0; b < batches; ++b)
    {
        for (int c = 0; c < input_channels; ++c)
        {
            for (int j = 0; j < input_height; ++j)
            {
                for (int i = 0; i < input_width; ++i)
                {
                    padded_input[b][c][j + padding][i + padding] = input[b][c][j][i];
                }
            }
        }
    }

    for (int b = 0; b < batches; b++)
    {
        for (int z = 0; z < output_channels; z++)
        {
            for (int y = 0; y < output_height; y++)
            {
                int base_input_j = y * stride; // Precompute row base index

                for (int x = 0; x < output_width; x++)
                {
                    int base_input_i = x * stride; // Precompute column base index
                    float sum = 0.0f;

                    for (int k = 0; k < input_channels; k++)
                    {
                        for (int j = 0; j < kernel_height; j++)
                        {
                            int input_j = base_input_j + j; // Add kernel row offset

                            for (int i = 0; i < kernel_width; i++)
                            {
                                int input_i = base_input_i + i; // Add kernel column offset
                                sum += padded_input[b][k][input_j][input_i] * kernel[z][k][j][i];
                            }
                        }
                    }
                    output[b][z][y][x] = sum;
                }
            }
        }
    }

    vector<float> output_flatten;
    for (int b = 0; b < batches; b++)
        for (int k = 0; k < output_channels; k++)
        {
            for (int j = 0; j < output_height; j++)
            {
                for (int i = 0; i < output_width; i++)
                {
                    output_flatten.push_back(output[b][k][j][i]);
                }
            }
        }

    write_to_binary("../outputs/chw_conv3d_cpp.bin", output_flatten);
}

int main()
{
    vector<int> input_dims = {1, 3, 224, 224};
    vector<int> kernel_dims = {64, 3, 7, 7};
    // int input_dim = 10;
    // int filter_dim = 4;

    auto input = read_npy_file("../inputs/py_input.npy", input_dims);
    auto kernel = read_npy_file("../weights/py_wt.npy", kernel_dims);

    int padding = 3;
    int stride = 2;

    Vector4D output;

    chw_conv3d(input, kernel, output, padding, stride);

    return 0;
}
