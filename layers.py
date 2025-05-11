import numpy as np
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNN, self).__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels

        scale = np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = scale * np.random.randn(
            out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
        )
        self.bias = np.zeros(out_channels)

        self.training = True

    def _pad_input(self, x):
        if self.padding[0] == 0 and self.padding[1] == 0:
            return x

        pad_width = (
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[0]),
            (self.padding[1], self.padding[1]),
        )
        return np.pad(x, pad_width, mode="constant", constant_values=0)

    def _im2col(self, x_padded, h_out, w_out):
        batch_size, in_channels, h_in, w_in = x_padded.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        # Initialize output matrix
        col = np.zeros(
            (batch_size * h_out * w_out, in_channels * k_h * k_w), dtype=x_padded.dtype
        )

        # Fill the column matrix
        idx = 0
        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * s_h
                    w_start = j * s_w
                    # Extract patch and flatten it
                    patch = x_padded[
                        b, :, h_start : h_start + k_h, w_start : w_start + k_w
                    ]
                    col[idx] = patch.reshape(-1)
                    idx += 1

        return col

    def forward(self, x):
        print(f"-------CNN forward: input shape: {x.shape}")
        is_torch_tensor = False
        if hasattr(x, "numpy"):
            is_torch_tensor = True
            x_np = x.detach().numpy()
        else:
            x_np = x

        batch_size, in_channels, h_in, w_in = x_np.shape

        assert in_channels == self.in_channels, (
            f"Expected input with {self.in_channels} channels, got {in_channels}"
        )

        x_padded = self._pad_input(x_np)
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        weights_reshaped = self.weight.reshape(self.out_channels, -1)

        x_col = self._im2col(x_padded, h_out, w_out)

        output = x_col @ weights_reshaped.T

        output = output + self.bias

        output = output.reshape(batch_size, h_out, w_out, self.out_channels)
        output = output.transpose(0, 3, 1, 2)

        if is_torch_tensor:
            import torch

            output = torch.from_numpy(output.astype(np.float32))

        return output

    def __call__(self, x):
        return self.forward(x)


class max_pooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(max_pooling, self).__init__()
        # Handle kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Handle stride
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def _pad_input(self, x):
        if self.padding[0] == 0 and self.padding[1] == 0:
            return x

        pad_width = (
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[0]),
            (self.padding[1], self.padding[1]),
        )
        return np.pad(x, pad_width, mode="constant", constant_values=0)

    def forward(self, x):
        print(f"-------max pooling forward: input shape: {x.shape}")
        # Convert PyTorch tensor to NumPy if necessary
        is_torch_tensor = False
        if hasattr(x, "numpy"):
            is_torch_tensor = True
            x_np = x.detach().numpy()
        else:
            x_np = x

        batch_size, channels, h_in, w_in = x_np.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        # Apply padding
        x_padded = self._pad_input(x_np)

        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding[0] - k_h) // s_h + 1
        w_out = (w_in + 2 * self.padding[1] - k_w) // s_w + 1

        # Initialize output
        output = np.zeros((batch_size, channels, h_out, w_out), dtype=x_np.dtype)

        # More efficient pooling implementation with fewer operations
        for b in range(batch_size):
            for c in range(channels):
                for i in range(h_out):
                    h_start = i * s_h
                    h_end = min(h_start + k_h, h_in + 2 * self.padding[0])

                    for j in range(w_out):
                        w_start = j * s_w
                        w_end = min(w_start + k_w, w_in + 2 * self.padding[1])

                        output[b, c, i, j] = np.max(
                            x_padded[b, c, h_start:h_end, w_start:w_end]
                        )

        # Convert back to PyTorch tensor if input was a PyTorch tensor
        if is_torch_tensor:
            import torch

            output = torch.from_numpy(output.astype(np.float32))

        return output

    def __call__(self, x):
        return self.forward(x)


class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace  # For PyTorch compatibility

    def forward(self, x):
        print(f"-------ReLU forward: input shape: {x.shape}")
        is_torch_tensor = False
        if hasattr(x, "numpy"):
            is_torch_tensor = True
            x_np = x.detach().numpy()
        else:
            x_np = x

        output = np.maximum(0, x_np)

        if is_torch_tensor:
            import torch

            output = torch.from_numpy(output.astype(np.float32))

        return output

    def __call__(self, x):
        return self.forward(x)
