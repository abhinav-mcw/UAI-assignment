from torch import nn
import torch
import numpy as np

def write_binary_file(file_path, array):
    with open(file_path, 'wb') as f:
        array.tofile(f)

def write_npy_file(file_path, array):
    np.save(file_path, array)

def convert_nchwc_to_nchw(filepath, input_tensor):
    input_nchw = input_tensor.permute(0, 2, 3, 1)  # NHWC to NCHW
    np.save(filepath, input_nchw.detach().numpy())

def convert_kernel_nchwc_to_nchw(filepath, kernel_tensor):
    kernel_nchw = kernel_tensor.permute(2, 3, 1, 0)  # NHWC to NCHW
    np.save(filepath, kernel_nchw.detach().numpy())

if __name__ == "__main__":
    # Specify dimensions in NCHW format for now
    input_dims = [1, 3, 224, 224]  # NCHW format (1, 3, 224, 224)
    kernel_dims = [64, 3, 7, 7]  # NCHW format (64, 3, 7, 7)

    input_height = input_dims[2]
    input_width = input_dims[3]
    input_channel = input_dims[1]

    kernel_height = kernel_dims[2]
    kernel_width = kernel_dims[3]
    kernel_channel = kernel_dims[1]

    stride = 2
    padding = 3
    output_channel = kernel_dims[0]

    # This flag will be used to indicate whether input is in NHWC format
    
    input_format = "NCHW"  

    # input_format = "NHWC"  


    # Load input and kernel data (example: adjust paths as needed)

    input_data = np.load("../inputs/py_input.npy")  # for nchw

    # input_data = np.load("../inputs/py_input_hwc.npy")  # for nhwc

    kernel_data = np.load("../weights/py_wt.npy")   # for nchw

    # kernel_data = np.load("../weights/py_hwc_wt.npy") # for nhwc


    if input_format == "NHWC":
        # Convert NHWC (1, 224, 224, 3) to NCHW (1, 3, 224, 224) for PyTorch
        input_tensor = torch.from_numpy(input_data).permute(0, 3, 1, 2)  # NHWC -> NCHW
        # Convert kernel from NHWC (7, 7, 3, 64) to NCHW (64, 3, 7, 7)
        kernel_tensor = torch.from_numpy(kernel_data).permute(3, 2, 0, 1)  # NHWC -> NCHW
    else:
        # No permutation needed for NCHW format
        input_tensor = torch.from_numpy(input_data)  # Already in NCHW format
        kernel_tensor = torch.from_numpy(kernel_data)  # Already in NCHW format

    print(input_tensor.shape)
    print(kernel_tensor.shape)
    # Define the convolutional layer
    conv = nn.Conv2d(
        in_channels=input_channel,  # 3
        out_channels=output_channel,  # 64
        kernel_size=(kernel_height, kernel_width),  # (7, 7)
        stride=stride,  # 2
        padding=padding  # 3
    )

    # Assign weights and biases
    bias_matrix = [0] * output_channel
    bias_tensor = torch.Tensor(bias_matrix)
    
    conv.weight.data = kernel_tensor
    conv.bias.data = bias_tensor

    # Perform the convolution
    output = conv(input_tensor)
    
    # Output as NCHW format
    output_np = output.detach().numpy()
    output_1d = output_np.flatten()
    write_binary_file('../outputs/py_chw_conv3d_output.bin', output_1d)

    # If the input format was NHWC, convert the output back to NHWC for saving
    if input_format == "NHWC":
        output_nhwc = output.permute(0, 2, 3, 1)  # Convert NCHW -> NHWC
        output_np = output_nhwc.detach().numpy()
        output_nhwc_1d = output_np.flatten()
        write_binary_file('../outputs/py_hwc_conv3d_output.bin', output_nhwc_1d)

    print("Python computation complete.")
    print("\t3D Convolution output written to 'py_chw_conv3d_output.bin'.")

    if input_format == "NHWC":
        print("\t3D Convolution output written to 'py_hwc_conv3d_output.bin'.")
