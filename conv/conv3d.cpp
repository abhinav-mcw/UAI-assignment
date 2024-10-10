#include <iostream>
#include <vector>
using namespace std;

void conv3d(int *input, int *filter, int *output, int input_dim, int filter_dim, int output_dim, int stride, int padding)
{
    int padded_input_dim = input_dim + 2 * padding;

    vector<int> padded_input(padded_input_dim * padded_input_dim * padded_input_dim, 0);

    // Copy original input into padded input
    for (int z = 0; z < input_dim; z++)
    {
        for (int y = 0; y < input_dim; y++)
        {
            for (int x = 0; x < input_dim; x++)
            {
                int inputIdx = z * input_dim * input_dim + y * input_dim + x;
                int paddedIdx = (z + padding) * padded_input_dim * padded_input_dim + (y + padding) * padded_input_dim + (x + padding);
                padded_input[paddedIdx] = input[inputIdx];
            }
        }
    }

    // Perform convolution with stride
    for (int z = 0; z < output_dim; z++)
    {
        for (int y = 0; y < output_dim; y++)
        {
            for (int x = 0; x < output_dim; x++)
            {
                long sum = 0;
                for (int k = 0; k < filter_dim; k++)
                {
                    for (int j = 0; j < filter_dim; j++)
                    {
                        for (int i = 0; i < filter_dim; i++)
                        {
                            int inputIdx = (z * stride + k) * padded_input_dim * padded_input_dim + (y * stride + j) * padded_input_dim + (x * stride + i);
                            int filterIdx = k * filter_dim * filter_dim + j * filter_dim + i;
                            sum += padded_input[inputIdx] * filter[filterIdx];
                        }
                    }
                }
                int outputIdx = z * output_dim * output_dim + y * output_dim + x;
                output[outputIdx] = sum;
            }
        }
    }
}

int main()
{
    const int input_dim = 4;
    const int filter_dim = 2;

    const int stride = 2;
    const int padding = 1;

    int output_dim = (input_dim - filter_dim + 2 * padding) / stride + 1;

    int input[input_dim * input_dim * input_dim] = {1, 0, 1, 0, 1, 1, 3, 1, 1, 1, 0, 2, 0, 2, 1, 1,
                                                    1, 0, 0, 1, 2, 0, 1, 2, 3, 1, 1, 1, 0, 0, 3, 1,
                                                    2, 0, 1, 1, 3, 3, 1, 0, 2, 1, 1, 0, 3, 2, 1, 2,
                                                    1, 0, 2, 0, 1, 0, 3, 3, 3, 1, 0, 0, 1, 1, 0, 2};

    int filter[filter_dim * filter_dim * filter_dim] = {1, 1, 0, 0, 2, 1, 0, 0};

    int output[output_dim * output_dim * output_dim];

    for (int z = 0; z < input_dim; z++)
    {
        for (int y = 0; y < input_dim; ++y)
        {
            for (int x = 0; x < input_dim; ++x)
            {
                std::cout << input[z * input_dim * input_dim + y * input_dim + x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << std::endl;

    conv3d(input, filter, output, input_dim, filter_dim, output_dim, stride, padding);

    for (int z = 0; z < output_dim; z++)
    {
        for (int y = 0; y < output_dim; ++y)
        {
            for (int x = 0; x < output_dim; ++x)
            {
                std::cout << output[z * output_dim * output_dim + y * output_dim + x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}