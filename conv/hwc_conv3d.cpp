
#include <iostream>
#include <vector>
#include "../utils/utils.hpp"

using namespace std;

using Vector4D = vector<vector<vector<vector<float>>>>;

void hwc_conv3d(const Vector4D &input,
                const Vector4D &kernel,
                Vector4D &output,
                int padding, int stride)
{
    int batches = input.size();
    int input_channels = input[0][0][0].size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();

    int output_channels = kernel[0][0][0].size();
    int kernel_height = kernel.size();
    int kernel_width = kernel[0].size();

    cout << input_height << " " << input_width << " " << input_channels << endl;

    cout << kernel_height << " " << kernel_width << " " << output_channels << endl;

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    cout << output_height << " " << output_width << " " << output_channels << endl;

    output.assign(batches, vector<vector<vector<float>>>(
                               output_height, vector<vector<float>>(
                                                  output_width, vector<float>(output_channels, 0))));

    vector<vector<vector<vector<float>>>>
        padded_input(batches, vector<vector<vector<float>>>(input_height + 2 * padding, vector<vector<float>>(input_width + 2 * padding, vector<float>(input_channels, 0))));

    // // Copy input to padded input
    for (int b = 0; b < batches; ++b)
    {
        for (int j = 0; j < input_height; ++j)
        {
            for (int i = 0; i < input_width; ++i)
            {
                for (int c = 0; c < input_channels; ++c)
                {
                    padded_input[b][j + padding][i + padding][c] = input[b][j][i][c];
                }
            }
        }
    }

    // // // Perform the convolution
    for (int b = 0; b < batches; b++)
    {
        for (int j = 0; j < output_height; j++)
        {
            int base_input_j = j * stride; // Precompute row base index for input

            for (int i = 0; i < output_width; i++)
            {
                int base_input_i = i * stride; // Precompute column base index for input

                for (int k = 0; k < output_channels; k++)
                {
                    float sum = 0.0f;

                    for (int y = 0; y < kernel_height; y++)
                    {
                        int input_j = base_input_j + y; // Add kernel row offset

                        for (int x = 0; x < kernel_width; x++)
                        {
                            int input_i = base_input_i + x; // Add kernel column offset

                            for (int z = 0; z < input_channels; z++)
                            {
                                sum += padded_input[b][input_j][input_i][z] * kernel[y][x][z][k];
                            }
                        }
                    }
                    output[b][j][i][k] = sum;
                }
            }
        }
    }

    vector<float> output_flatten;
    for (int b = 0; b < batches; b++)

        for (int j = 0; j < output_height; j++)
        {
            for (int i = 0; i < output_width; i++)
                for (int k = 0; k < output_channels; k++)
                {
                    {
                        output_flatten.push_back(output[b][j][i][k]);
                    }
                }
        }

    write_to_binary("../outputs/hwc_conv3d_cpp.bin", output_flatten);
}

int main()
{
    vector<int> input_dims = {1, 224, 224, 3};
    vector<int> kernel_dims = {7, 7, 3, 64};
    auto input = read_npy_file("../inputs/py_input_hwc.npy", input_dims);
    auto kernel = read_npy_file("../weights/py_conv_hwc_wt.npy", kernel_dims);

    Vector4D output;

    int padding = 3;
    int stride = 2;
    // Perform the convolution
    hwc_conv3d(input, kernel, output, padding, stride);

    return 0;
}
