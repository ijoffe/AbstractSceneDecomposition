# Isaac Joffe and Benjamin Colussi, 2025


# Usage:
# python monet_normask.py --n MONet_MultiDSprites_NoRMask_V0 --m Train --d MultiDSprites --s 5 --b 128 --e 1000 --l 0.0001
# python monet_normask.py --n MONet_ObjectsRoom_NoRMask_V0 --m Train --d ObjectsRoom --s 7 --b 64 --e 1000 --l 0.0001
# python monet_normask.py --n MONet_Tetrominoes_NoRMask_V0 --m Train --d Tetrominoes --s 4 --b 128 --e 1000 --l 0.0001


# Fundamental PyTorch utilities to build model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
# Libraries to use datasets
from multi_object_datasets_torch import ClevrWithMasks, MultiDSprites, ObjectsRoom, Tetrominoes
from arc_data import ARCAGI
# Additional common libraries
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm
import argparse


# Get GPU information
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}")
if device == "cuda":
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True


"""
VAE of the system.
"""
class VAE_NoRMask(nn.Module):
    """
    Creates the VAE, building the encoder and decoder.
    """
    def __init__(self, batch_size):
        super().__init__()

        # Pass along batch size to ensure dimensions are consistent
        self.batch_size = batch_size

        # Encoder of the VAE
        # "It receives the concatenation of the input image x ...
        #     and the attention mask in logarithmic units, log mk as input."
        self.encoder_nn = nn.Sequential(
            # "The VAE encoder is a standard CNN with 3x3 kernels, stride 2, and ReLU activations."
            # "The CNN layers output (32, 32, 64, 64) channels respectively."
            # Layer #1
            nn.Conv2d(              # "convolution"
                in_channels=4,      # "the concatenation of the input image x and the attention mask"
                out_channels=32,    # "32"
                kernel_size=3,      # "3x3 kernels"
                stride=2,           # "stride 2"
                padding=1,          # ASSUMPTION
                bias=True,          # ASSUMPTION
            ),                      # 32*32 output because I=64, K=3, S=2, P=1, floor((64-3+2(1))/2)+1=32
            nn.ReLU(),              # "ReLU activations"
            # Layer #2
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=32,    # "32"
                kernel_size=3,      # "3x3 kernels"
                stride=2,           # "stride 2"
                padding=1,          # ASSUMPTION
                bias=True,          # ASSUMPTION
            ),                      # 16*16 output because I=32, K=3, S=2, P=1, floor((32-3+2(1))/2)+1=16
            nn.ReLU(),              # "ReLU activations"
            # Layer #3
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=64,    # "64"
                kernel_size=3,      # "3x3 kernels"
                stride=2,           # "stride 2"
                padding=1,          # ASSUMPTION
                bias=True,          # ASSUMPTION
            ),                      # 8*8 output because I=16, K=3, S=2, P=1, floor((16-3+2(1))/2)+1=8
            nn.ReLU(),              # "ReLU activations"
            # Layer #4
            nn.Conv2d(              # "convolution"
                in_channels=64,     # FROM PREVIOUS LAYER
                out_channels=64,    # "64"
                kernel_size=3,      # "3x3 kernels"
                stride=2,           # "stride 2"
                padding=1,          # ASSUMPTION
                bias=True,          # ASSUMPTION
            ),                      # 4*4 output because I=8, K=3, S=2, P=1, floor((8-3+2(1))/2)+1=4
            nn.ReLU(),              # "ReLU activations"
            # "The CNN output is flattened and fed to a 2 layer MLP with output sizes of (256, 32)."
            # "The MLP output parameterises the μ and log σ of a 16-dim Gaussian latent posterior."
            nn.Flatten(),
            nn.Linear(              # "MLP"
                in_features=1024,   # FROM PREVIOUS LAYER
                out_features=256,   # "256"
            ),                      # ASSUMPTION (no activation)
            nn.Linear(              # "MLP"
                in_features=256,    # FROM PREVIOUS LAYER
                out_features=32,    # "32"
            ),                      # ASSUMPTION (no activation)
        )

        # Decoder of the VAE
        self.decoder_nn = nn.Sequential(
            # "The input to the broadcast decoder is a spatial tiling of zk concatenated with ...
            #     a pair of coordinate channels – one for each spatial dimension – ranging from -1 to 1."
            # "These go through a four-layer CNN with no padding, 3x3 kernels, ...
            #     stride 1, 32 output channels and ReLU activations."
            # Layer #1
            nn.Conv2d(              # "convolution"
                in_channels=18,     # "a spatial tiling of zk concatenated with a pair of coordinate channels"
                out_channels=32,    # "32 output channels"
                kernel_size=3,      # "3x3 kernels"
                stride=1,           # "stride 1"
                padding=0,          # "no padding"
                bias=True,          # ASSUMPTION
            ),                      # 70*70 output because I=72, K=3, S=1, P=0, floor((72-3+2(0))/1)+1=70
            nn.ReLU(),              # "ReLU activations"
            # Layer #2
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=32,    # "32 output channels"
                kernel_size=3,      # "3x3 kernels"
                stride=1,           # "stride 1"
                padding=0,          # "no padding"
                bias=True,          # ASSUMPTION
            ),                      # 68*68 output because I=70, K=3, S=1, P=0, floor((70-3+2(0))/1)+1=68
            nn.ReLU(),              # "ReLU activations"
            # Layer #3
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=32,    # "32 output channels"
                kernel_size=3,      # "3x3 kernels"
                stride=1,           # "stride 1"
                padding=0,          # "no padding"
                bias=True,          # ASSUMPTION
            ),                      # 66*66 output because I=68, K=3, S=1, P=0, floor((68-3+2(0))/1)+1=66
            nn.ReLU(),              # "ReLU activations"
            # Layer #4
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=32,    # "32 output channels"
                kernel_size=3,      # "3x3 kernels"
                stride=1,           # "stride 1"
                padding=0,          # "no padding"
                bias=True,          # ASSUMPTION
            ),                      # 64*64 output because I=66, K=3, S=1, P=0, floor((66-3+2(0))/1)+1=64
            nn.ReLU(),              # "ReLU activations"
            # Remove reconstructed mask from output (CHANGE)
            # "A final 1x1 convolutional layer transforms the output into 4 channels: ...
            #     3 RGB channels for the means of the image components xˆk, and ...
            #     1 for the logits used for the softmax operation to compute the reconstructed attention masks mˆk."
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=3,     # CHANGE
                kernel_size=1,      # "1x1 convolutional layer"
                stride=1,           # ASSUMPTION
                padding=0,          # ASSUMPTION
                bias=True,          # ASSUMPTION
            ),                      # 64*64 output because I=64, K=1, S=1, P=0, floor((64-1+2(0))/1)+1=64
        )
        return
        
    """
    Uses the encoder of the VAE to generate the latent distribution.
        Inputs: 64*64 RGB image (x), 64*64 logarithmic mask (log_mk)
        Outputs: 16-dimensional Gaussian latent posterior (mu, log_sig)
    """
    def encode(self, x, log_mk):
        # Encode the input into a latent representation
        # "It receives the concatenation of the input image x and ...
        #     the attention mask in logarithmic units, log mk as input."
        latent_repr = self.encoder_nn(torch.concat((x, log_mk), dim=1))
        # assert (len(latent_repr.shape) == 2) and (latent_repr.shape[0] == self.batch_size) and (latent_repr.shape[1] == 32)

        # Convert this latent representation into the probability distribution
        # "The MLP output parameterises the μ and logσ of a 16-dim Gaussian latent posterior."
        mu = torch.split(latent_repr, 16, dim=1)[0]
        log_sig = torch.split(latent_repr, 16, dim=1)[1]
        # assert (len(mu.shape) == 2) and (mu.shape[0] == self.batch_size) and (mu.shape[1] == 16)
        # assert (len(log_sig.shape) == 2) and (log_sig.shape[0] == self.batch_size) and (log_sig.shape[1] == 16)

        # Output of the encoder is the parameters of the probability distribution
        return mu, log_sig
    
    """
    Samples the latent distribution to generate a latent representation.
        Inputs: 16-dimensional Gaussian latent posterior (mu, log_sig)
        Outputs: Sampled latent vector (z)
    """
    def reparameterize(self, mu, log_sig):
        # Sample the represented distribution based on its mean and standard deviation
        std = torch.exp(log_sig)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # assert (len(z.shape) == 2) and (z.shape[0] == self.batch_size) and (z.shape[1] == 16)

        # Output of reparameterization is sampled latent vector
        return z
    
    """
    Uses the decoder of the VAE to reconstruct a component of the image and the mask.
        Inputs: 72*72*18 broadcasted sampled representation
        Outputs: 64*64 RGB reconstructed image component means (reconstructed_repr)
    """
    def decode(self, x):
        # Decode the output from a latent representation
        # Convert this output into the reconstructed image but no mask (CHANGE)
        # "3 RGB channels for the means of the image components xˆk, and 1 for the logits ...
        #     used for the softmax operation to compute the reconstructed attention masks mˆk"
        reconstructed_repr = self.decoder_nn(x)
        # assert (len(reconstructed_repr.shape) == 4) and (reconstructed_repr.shape[0] == self.batch_size) and (reconstructed_repr.shape[1] == 3) and (reconstructed_repr.shape[2] == 64) and (reconstructed_repr.shape[3] == 64)

        # Output of the decoder is the reconstructed image component means but no mask (CHANGE)
        return reconstructed_repr
    
    """
    Performs a full forward pass of the VAE, including both encoding and decoding.
        Inputs: 64*64 RGB image (x), 64*64 logarithmic mask (log_mk)
        Outputs: 16-dimensional Gaussian latent posterior (mu, log_sig), 64*64 RGB reconstructed image ...
            component means (x_hat_means)
    """
    def forward(self, x, log_mk):
        # First, encode the data into the latent space
        mu, log_sig = self.encode(x, log_mk)
        # assert (len(mu.shape) == 2) and (mu.shape[0] == self.batch_size) and (mu.shape[1] == 16)
        # assert (len(log_sig.shape) == 2) and (log_sig.shape[0] == self.batch_size) and (log_sig.shape[1] == 16)

        # Second, transform the latent distributions into a sampled image
        # "The input to the broadcast decoder is a spatial tiling of zk ...
        #     concatenated with a pair of coordinate channels - one for ...
        #     each spatial dimension - ranging from -1 to 1."
        z = self.reparameterize(mu, log_sig)
        # assert (len(z.shape) == 2) and (z.shape[0] == self.batch_size) and (z.shape[1] == 16)
        z = z.reshape((self.batch_size, 16, 1, 1)).repeat((1, 1, 72, 72))
        # assert (len(z.shape) == 4) and (z.shape[0] == self.batch_size) and (z.shape[1] == 16) and (z.shape[2] == 72) and (z.shape[3] == 72)
        dim1 = torch.linspace(-1, 1, 72, device=device)
        dim2 = torch.linspace(-1, 1, 72, device=device)
        dim1, dim2 = torch.meshgrid(dim1, dim2, indexing="ij")
        dim1 = dim1.reshape((1, 1, 72, 72)).repeat((self.batch_size, 1, 1, 1))
        dim2 = dim2.reshape((1, 1, 72, 72)).repeat((self.batch_size, 1, 1, 1))
        # assert (len(dim1.shape) == 4) and (dim1.shape[0] == self.batch_size) and (dim1.shape[1] == 1) and (dim1.shape[2] == 72) and (dim1.shape[3] == 72)
        # assert (len(dim2.shape) == 4) and (dim2.shape[0] == self.batch_size) and (dim2.shape[1] == 1) and (dim2.shape[2] == 72) and (dim2.shape[3] == 72)

        # Third, decode the data from the latent space
        x_hat_means = self.decode(torch.concat((z, dim1, dim2), dim=1))
        # assert (len(x_hat_means.shape) == 4) and (x_hat_means.shape[0] == self.batch_size) and (x_hat_means.shape[1] == 3) and (x_hat_means.shape[2] == 64) and (x_hat_means.shape[3] == 64)

        # Output of the overall VAE is the parameters of the probability distribution and the reconstructed image and mask
        return mu, log_sig, x_hat_means


"""
Attention network of the system.
"""
class Attention_NoRMask(nn.Module):
    """
    Creates the attention network, building the downwards and upwards paths of the UNet.
    """
    def __init__(self, batch_size):
        super().__init__()
        
        # Pass along batch size to ensure dimensions are consistent
        self.batch_size = batch_size

        # Downsampling path of the U-Net
        # "We used a standard U-Net blueprint with five blocks each on the downsampling and upsampling paths."
        # "At the kth attention step, the attention network receives the concatenation of the input image x ...
        #     and the current scope mask in log units, logsk, as input."
        # "Each block consists of the following: a 3x3 bias-free convolution with stride 1, ...
        #     followed by instance normalisation with a learned bias term, followed by ...
        #     a ReLU activation, and finally downsampled or upsampled by a factor of 2 using ...
        #     nearest neighbour-resizing (no resizing occurs in the last block of each path)."
        # Block #1
        self.down_nn_1 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=4,      # "the concatenation of the input image x and the current scope mask"
                out_channels=8,     # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 64*64 output because I=64, K=3, S=1, P=1, floor((64-3+2(1))/1)+1=64
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=8,     # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 64*64 output maintained
            nn.ReLU(),              # "ReLU activation"
        )
        self.down_sample_1 = nn.Sequential(
            nn.AvgPool2d(           # "and finally downsampled or upsampled"
                kernel_size=2,      # "by a factor of 2"
                stride=2,           # "by a factor of 2"
                padding=0,          # ASSUMPTION
                ceil_mode=True,     # ASSUMPTION
            ),                      # 32*32 output because I=64, K=2, S=2, P=0, ceil((64-2+2(0))/2)+1=32
        )
        # Block #2
        self.down_nn_2 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=8,      # FROM PREVIOUS LAYER
                out_channels=16,    # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 32*32 output because I=32, K=3, S=1, P=1, floor((32-3+2(1))/1)+1=32
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=16,    # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 32*32 output maintained
            nn.ReLU(),              # "ReLU activation"
        )
        self.down_sample_2 = nn.Sequential(
            nn.AvgPool2d(           # "and finally downsampled or upsampled"
                kernel_size=2,      # "by a factor of 2"
                stride=2,           # "by a factor of 2"
                padding=0,          # ASSUMPTION
                ceil_mode=True,     # ASSUMPTION
            ),                      # 16*16 output because I=29, K=2, S=2, P=0, ceil((32-2+2(0))/2)+1=16
        )
        # Block #3
        self.down_nn_3 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=16,     # FROM PREVIOUS LAYER
                out_channels=32,    # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 16*16 output because I=16, K=3, S=1, P=1, floor((16-3+2(1))/1)+1=16
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=32,    # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 16*16 output maintained
            nn.ReLU(),              # "ReLU activation"
        )
        self.down_sample_3 = nn.Sequential(
            nn.AvgPool2d(           # "and finally downsampled or upsampled"
                kernel_size=2,      # "by a factor of 2"
                stride=2,           # "by a factor of 2"
                padding=0,          # ASSUMPTION
                ceil_mode=True,     # ASSUMPTION
            ),                      # 8*8 output because I=16, K=2, S=2, P=0, ceil((16-2+2(0))/2)+1=8
        )
        # Block #4
        self.down_nn_4 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=64,    # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 8*8 output because I=8, K=3, S=1, P=1, floor((8-3+2(1))/1)+1=8
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=64,    # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 8*8 output maintained
            nn.ReLU(),              # "ReLU activation"
        )
        self.down_sample_4 = nn.Sequential(
            nn.AvgPool2d(           # "and finally downsampled or upsampled"
                kernel_size=2,      # "by a factor of 2"
                stride=2,           # "by a factor of 2"
                padding=0,          # ASSUMPTION
                ceil_mode=True,     # ASSUMPTION
            ),                      # 4*4 output because I=5, K=2, S=2, P=0, ceil((8-2+2(0))/2)+1=4
        )
        # Block #5
        self.down_nn_5 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=64,     # FROM PREVIOUS LAYER
                out_channels=128,   # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 4*4 output because I=4, K=3, S=1, P=1, floor((4-3+2(1))/1)+1=4
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=128,   # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 4*4 output maintained
            nn.ReLU(),              # "ReLU activation"
            # "no resizing occurs in the last block of each path"
        )

        # Skip connections of the UNet
        # "Skip tensors are collected from each block in the downsampling path ...
        #     after the ReLU activation function. These are concatenated with input ...
        #     tensors along the upsampling blocks before the convolutional layer."
        self.skip_nn = nn.Sequential(
            # Implemented in forward() function
            None,
        )

        # Nonskip connections of the UNet
        # # "A 3-layer MLP serves as the non-skip connection between the downsampling and ...
        #     upsampling paths with its final output dimensionally matching that of the ...
        #     last skip tensor.""
        self.middle_nn = nn.Sequential(
            # "The intermediate hidden layers were sized (128, 128). The input to the MLP is ...
            #     the last skip tensor collected from the downsampling path (after flattening). ...
            #     A ReLU activation is applied after all three output layers. The final output is ...
            #     then reshaped to match that of the last skip tensor, concatenated with it, ...
            #     and finally fed into the upsampling path."
            nn.Flatten(),
            # Layer #1
            nn.Linear(              # "MLP"
                in_features=2048,   # FROM PREVIOUS LAYER
                out_features=128,   # "128"
            ),
            nn.ReLU(),              # "ReLU activation"
            # Layer #2
            nn.Linear(              # "MLP"
                in_features=128,    # FROM PREVIOUS LAYER
                out_features=128,   # "128"
            ),
            nn.ReLU(),              # "ReLU activation"
            # Layer #3
            nn.Linear(              # "MLP"
                in_features=128,    # FROM PREVIOUS LAYER
                out_features=2048,  # "reshaped to match that of the last skip tensor"
            ),
            nn.ReLU(),              # "ReLU activation"
        )

        # "We used a standard U-Net blueprint with five blocks each on the downsampling and upsampling paths."
        # "Each block consists of the following: a 3x3 bias-free convolution with stride 1, ...
        #     followed by instance normalisation with a learned bias term, followed by ...
        #     a ReLU activation, and finally downsampled or upsampled by a factor of 2 using ...
        #     nearest neighbour-resizing (no resizing occurs in the last block of each path)."
        # Block #1
        self.up_nn_1 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=256,    # FROM PREVIOUS LAYER
                out_channels=64,    # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 4*4 output because I=4, K=3, S=1, P=1, floor((4-3+2(1))/1)+1=4
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=64,    # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 4*4 output maintained
            nn.ReLU(),              # "ReLU activation"
            nn.Upsample(            # "and finally downsampled or upsampled"
                scale_factor=2,     # "by a factor of 2"
                mode="nearest",     # "using nearest neighbour-resizing"
            ),
        )
        # Block #2
        self.up_nn_2 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=128,    # FROM PREVIOUS LAYER
                out_channels=32,    # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 8*8 output because I=8, K=3, S=1, P=1, floor((8-3+2(1))/1)+1=8
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=32,    # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 8*8 output maintained
            nn.ReLU(),              # "ReLU activation"
            nn.Upsample(            # "and finally downsampled or upsampled"
                scale_factor=2,     # "by a factor of 2"
                mode="nearest",     # "using nearest neighbour-resizing"
            ),
        )
        # Block #3
        self.up_nn_3 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=64,     # FROM PREVIOUS LAYER
                out_channels=16,    # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 16*16 output because I=16, K=3, S=1, P=1, floor((16-3+2(1))/1)+1=16
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=16,    # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 16*16 output maintained
            nn.ReLU(),              # "ReLU activation"
            nn.Upsample(            # "and finally downsampled or upsampled"
                scale_factor=2,     # "by a factor of 2"
                mode="nearest",     # "using nearest neighbour-resizing"
            ),
        )
        # Block #4
        self.up_nn_4 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=32,     # FROM PREVIOUS LAYER
                out_channels=8,     # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 32*32 output because I=32, K=3, S=1, P=1, floor((32-3+2(1))/1)+1=32
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=8,     # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 32*32 output maintained
            nn.ReLU(),              # "ReLU activation"
            nn.Upsample(            # "and finally downsampled or upsampled"
                scale_factor=2,     # "by a factor of 2"
                mode="nearest",     # "using nearest neighbour-resizing"
            ),
        )
        # Block #5
        self.up_nn_5 = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=16,     # FROM PREVIOUS LAYER
                out_channels=4,     # ASSUMPTION
                kernel_size=3,      # "3x3 bias-free convolution"
                stride=1,           # "stride 1"
                padding=1,          # ASSUMPTION
                bias=False,         # "bias-free"
            ),                      # 64*64 output because I=64, K=3, S=1, P=1, floor((64-3+2(1))/1)+1=64
            nn.InstanceNorm2d(      # "instance normalisation"
                num_features=4,     # FROM PREVIOUS LAYER
                affine=True,        # "with a learned bias term"
            ),                      # 64*64 output maintained
            nn.ReLU(),              # "ReLU activation"
            # "no resizing occurs in the last block of each path"
        )

        # "Following the upsampling path, a final 1x1 convolution with stride 1 ...
        #     and a single output channel transforms the U-Net output into the ...
        #     logits for αk. Both log αk and log(1 − αk) are computed directly in ...
        #     log units from the logits (using the log softmax operation). Each are ...
        #     added to the current scope (also maintained in log units) log sk−1 to ...
        #     compute the next (log) attention mask log mk and next (log) scope log sk, respectively.""
        self.final_nn = nn.Sequential(
            nn.Conv2d(              # "convolution"
                in_channels=4,      # FROM PREVIOUS LAYER
                out_channels=1,     # "a single output channel"
                kernel_size=1,      # "1x1 convolution"
                stride=1,           # "stride 1"
                padding=0,          # ASSUMPTION
                bias=True,          # ASSUMPTION
            ),                      # 64*64 output because I=64, K=1, S=1, P=0, floor((64-1+2(0))/1)+1=64
        )
        return
    
    """
    Performs a full forward pass of the attention network, updating internal state.
        Inputs: 64*64 RGB image (x), current logarithmic scope (log_sk)
        Outputs: 64*64 logarithmic mask (log_mk), next logarithmic scope (log_skp1)
    """
    def forward(self, x, log_sk):
        # First, downsample the data down the "U" of the UNet
        x1 = self.down_nn_1(torch.concat((x, log_sk), dim=1))
        # assert (len(x1.shape) == 4) and (x1.shape[0] == self.batch_size) and (x1.shape[1] == 8) and (x1.shape[2] == 64) and (x1.shape[3] == 64)
        x2 = self.down_nn_2(self.down_sample_1(x1))
        # assert (len(x2.shape) == 4) and (x2.shape[0] == self.batch_size) and (x2.shape[1] == 16) and (x2.shape[2] == 32) and (x2.shape[3] == 32)
        x3 = self.down_nn_3(self.down_sample_2(x2))
        # assert (len(x3.shape) == 4) and (x3.shape[0] == self.batch_size) and (x3.shape[1] == 32) and (x3.shape[2] == 16) and (x3.shape[3] == 16)
        x4 = self.down_nn_4(self.down_sample_3(x3))
        # assert (len(x4.shape) == 4) and (x4.shape[0] == self.batch_size) and (x4.shape[1] == 64) and (x4.shape[2] == 8) and (x4.shape[3] == 8)
        x5 = self.down_nn_5(self.down_sample_4(x4))
        # assert (len(x5.shape) == 4) and (x5.shape[0] == self.batch_size) and (x5.shape[1] == 128) and (x5.shape[2] == 4) and (x5.shape[3] == 4)

        # Second, compute the nonskip connection at the bottom of the "U" of the UNet
        y0 = self.middle_nn(x5).reshape((self.batch_size, 128, 4, 4))
        # assert (len(y0.shape) == 4) and (y0.shape[0] == self.batch_size) and (y0.shape[1] == 128) and (y0.shape[2] == 4) and (y0.shape[3] == 4)

        # Third, upsample the data up the "U" of the UNet
        # Successively call upsampling networks on processed data
        # Apply skip connections by concatenating previous data to the current data
        y1 = self.up_nn_1(torch.concat((y0, x5), dim=1))
        # assert (len(y1.shape) == 4) and (y1.shape[0] == self.batch_size) and (y1.shape[1] == 64) and (y1.shape[2] == 8) and (y1.shape[3] == 8)
        y2 = self.up_nn_2(torch.concat((y1, x4), dim=1))
        # assert (len(y2.shape) == 4) and (y2.shape[0] == self.batch_size) and (y2.shape[1] == 32) and (y2.shape[2] == 16) and (y2.shape[3] == 16)
        y3 = self.up_nn_3(torch.concat((y2, x3), dim=1))
        # assert (len(y3.shape) == 4) and (y3.shape[0] == self.batch_size) and (y3.shape[1] == 16) and (y3.shape[2] == 32) and (y3.shape[3] == 32)
        y4 = self.up_nn_4(torch.concat((y3, x2), dim=1))
        # assert (len(y4.shape) == 4) and (y4.shape[0] == self.batch_size) and (y4.shape[1] == 8) and (y4.shape[2] == 64) and (y4.shape[3] == 64)
        y5 = self.up_nn_5(torch.concat((y4, x1), dim=1))
        # assert (len(y5.shape) == 4) and (y5.shape[0] == self.batch_size) and (y5.shape[1] == 4) and (y5.shape[2] == 64) and (y5.shape[3] == 64)

        # Fourth, compute the output processing at the end of the UNet
        y = self.final_nn(y5)
        # assert (len(y.shape) == 4) and (y.shape[0] == self.batch_size) and (y.shape[1] == 1) and (y.shape[2] == 64) and (y.shape[3] == 64)
        log_alpha = nn.LogSigmoid()(y)
        # assert (len(log_alpha.shape) == 4) and (log_alpha.shape[0] == self.batch_size) and (log_alpha.shape[1] == 1) and (log_alpha.shape[2] == 64) and (log_alpha.shape[3] == 64)

        # Fifth, translate the network output to the desired format
        # "The attention mask for step k is given by mk = sk−1αψ(x;sk−1)"
        log_mk = log_sk + log_alpha
        # assert (len(log_mk.shape) == 4) and (log_mk.shape[0] == self.batch_size) and (log_mk.shape[1] == 1) and (log_mk.shape[2] == 64) and (log_mk.shape[3] == 64)
        # "The scope for the next step is given by sk+1 = sk(1−αψ(x;sk))"
        log_skp1 = log_sk + log_alpha - y
        # assert (len(log_skp1.shape) == 4) and (log_skp1.shape[0] == self.batch_size) and (log_skp1.shape[1] == 1) and (log_skp1.shape[2] == 64) and (log_skp1.shape[3] == 64)

        # Output of the attention network is the mask at the current time step
        return log_mk, log_skp1


"""
Overall MONet model.
"""
class MONet_NoRMask(nn.Module):
    """
    Creates MONet, building the VAE and attention network.
    """
    def __init__(self, num_slots, batch_size, learning_rate):
        super().__init__()

        # Model general hyperparameters
        # "We used RMSProp for optimisation with a learning rate of 0.0001, and a batch size of 64."
        self.batch_size = batch_size             # "a batch size of 64"
        self.learning_rate = learning_rate       # "a learning rate of 0.0001"

        # Model construction hyperparameters
        # "We trained MONet with K=7 slots."
        self.K = num_slots    # "K=7 slots"
        # "The loss weights were β = 0.5, γ = 0.5."
        self.alpha = 1
        self.beta = 0.5       # "β = 0.5"
        # "For the MONet experiments, the first "background" component scale was ...
        #     fixed at σbg = 0.09, and for the K − 1 remaining "foreground" components, ...
        #     the scale was fixed at σfg = 0.11.
        self.sigma_bg = 0.09    # "σbg = 0.09"
        self.sigma_fg = 0.11    # "σfg = 0.11"

        # Model loss
        # First term represents the VAE image reconstruction loss (drives the decoder to properly reconstruct masked region)
        self.loss_1 = None
        # Second term represents the regularization of the VAE (drives the encoder to generate a normal distribution) weighted by beta
        self.loss_2 = None
        # Overall loss combines these two terms (CHANGE)
        self.loss = None

        # VAE of the model
        # "The component VAE is a neural network, with an encoder parameterised by φ and a decoder parameterised by θ."
        self.vae = VAE_NoRMask(self.batch_size)

        # Attention network of the model
        # "The mask distribution is learned by the attention module, a neural network conditioned on x and parameterised by ψ."
        self.attention = Attention_NoRMask(self.batch_size)

        # Model optimizer
        # "We used RMSProp for optimisation with a learning rate of 0.0001, and a batch size of 64."
        self.optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
        )
        return

    """
    Performs a full forward pass of MONet, utilizing multiple passes of the VAE and attention network.
        Inputs: 64*64 RGB image (x)
        Outputs: 16-dimensional Gaussian latent posteriors (mus, log_sigs), 64*64 logarithmic masks (log_masks), ...
            64*64 RGB reconstructed image components means (x_hat_means)
    """
    def forward(self, x):
        # The parameters of the latent distributions generated by the VAE
        mus = torch.zeros((self.K, self.batch_size, 16), device=device)
        log_sigs = torch.zeros((self.K, self.batch_size, 16), device=device)
        # The masks recurrently generated by the attention network
        log_masks = torch.zeros((self.K, self.batch_size, 1, 64, 64), device=device)
        # The current state of the model
        # Initialize recurrent state variable "with the first scope s0 = 1"
        log_states = torch.zeros((self.K, self.batch_size, 1, 64, 64), device=device)
        log_states[0] = torch.log(torch.ones((self.batch_size, 1, 64, 64), device=device))
        # The region of the image corresponding to the mask as reconstructed by the VAE
        recon_comp_means = torch.zeros((self.K, self.batch_size, 3, 64, 64), device=device)

        # Perform normal steps 1, ..., K-1
        for k in range(self.K - 1):
            # Mask k is simply the output of the attention network based on the image (and its current internal state)
            log_masks[k], log_states[k+1] = self.attention(x, log_states[k])
            # Component and mask reconstruction k is the output of the VAE based on the image and desired attention mask
            mus[k], log_sigs[k], recon_comp_means[k] = self.vae(x, log_masks[k])
        # Perform final step K, which is different
        # Mask K is the remaining scope to be explained, extracted directly from the attention network
        log_masks[self.K-1] = log_states[self.K-1]
        # Component and mask reconstruction K is still the normal output of the VAE
        mus[self.K-1], log_sigs[self.K-1], recon_comp_means[self.K-1] = self.vae(x, log_masks[self.K-1])

        # assert (len(mus.shape) == 3) and (mus.shape[0] == self.K) and (mus.shape[1] == self.batch_size) and (mus.shape[2] == 16)
        # assert (len(log_sigs.shape) == 3) and (log_sigs.shape[0] == self.K) and (log_sigs.shape[1] == self.batch_size) and (log_sigs.shape[2] == 16)
        # assert (len(log_masks.shape) == 5) and (log_masks.shape[0] == self.K) and (log_masks.shape[1] == self.batch_size) and (log_masks.shape[2] == 1) and (log_masks.shape[3] == 64) and (log_masks.shape[4] == 64)
        # assert (len(log_states.shape) == 5) and (log_states.shape[0] == self.K) and (log_states.shape[1] == self.batch_size) and (log_states.shape[2] == 1) and (log_states.shape[3] == 64) and (log_states.shape[4] == 64)
        # assert (len(recon_comp_means.shape) == 5) and (recon_comp_means.shape[0] == self.K) and (recon_comp_means.shape[1] == self.batch_size) and (recon_comp_means.shape[2] == 3) and (recon_comp_means.shape[3] == 64) and (recon_comp_means.shape[4] == 64)
        masks_sum = torch.sum(log_masks.exp()) / self.batch_size / 64 / 64
        if (masks_sum < 0.99) or (masks_sum > 1.01):
            print(f"WARNING: Mask distributions do not sum to 1. Computed value: {masks_sum}")

        # Output of MONet is the parameters of the probability distribution, mask, and reconstructed image and mask at the current time step
        return mus, log_sigs, log_masks, recon_comp_means
    
    """
    Train MONet.
        Inputs: Data to train on (dataloaders), number of iterations to train for (epochs), location where model should be stored (name)
        Outputs: Saved model weights
    """
    def learn(self, dataloaders, epochs, name):
        # Create directory to save model in, if it does not yet exist
        if not os.path.exists(f"models/{name}/"):
            os.makedirs(f"models/{name}/")

        # Set up optimized training
        scaler = torch.amp.GradScaler("cuda")

        losses = []
        # Iterate over each training epoch
        for epoch in tqdm(range(epochs)):
            start = time()
            losses.append([])
            # Iterate through each dataset to train on
            for dataloader in dataloaders:
                # Iterate through each batch of the dataset
                for i, x in enumerate(dataloader):
                    # Extract image from training data (unsupervised, so this is all that is needed)
                    image = (x["image"] / 255).to(device)
                    # Only deal with complete batches for simplicity's sake
                    if image.shape[0] != self.batch_size:
                        break

                    # Perform forward pass, computing model outputs in optimized manner
                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device):
                        # Perform complete forward pass of the complete model
                        mus, log_sigs, log_masks, recon_comp_means = self(image)

                        # Compute reconstruction loss term
                        # Image reconstruction standard deviations are uniform across images but different for foreground and background slots
                        recon_comp_sigmas = torch.Tensor([self.sigma_bg if k == 0 else self.sigma_fg for k in range(self.K)])
                        # Sum up contribution of each slot to the reconstruction loss
                        reconstruction_loss = 0
                        for j in range(0, self.K):
                            # Use derived formula to compute weighted difference
                            # "the VAE’s decoder likelihood term in the loss pθ(x|zk) is weighted according ...
                            #     to the mask, such that it is unconstrained outside of the masked regions."
                            reconstruction_loss = reconstruction_loss + torch.exp(log_masks[j] - torch.log(recon_comp_sigmas[j]) - 0.5 * (image - recon_comp_means[j]).pow(2) / recon_comp_sigmas[j].pow(2))
                        # Negative log operation required to convert to proper loss function, summed across all pixels
                        reconstruction_loss = torch.sum(-torch.log(reconstruction_loss))
                        # Weight loss term by alpha hyperparameter
                        self.loss_1 = self.alpha * reconstruction_loss / self.batch_size

                        # Compute VAE KL divergence loss term
                        # Each encoded representation is independent, so the KL divergence is additive
                        kld_loss = 0
                        for j in range(0, self.K):
                            # Use closed-form expression from class to compute KL divergence
                            kld_loss = kld_loss + torch.sum(torch.exp(log_sigs[j]).pow(2) + mus[j].pow(2) - 2 * log_sigs[j] - 1) / 2
                        # Weight loss term by beta hyperparameter
                        self.loss_2 = self.beta * kld_loss / self.batch_size

                        # Compute overall loss
                        self.loss = self.loss_1 + self.loss_2
                    
                    # Perform backward pass, gradient descent update on all parameters
                    scaler.scale(self.loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
                    scaler.step(self.optimizer)
                    scaler.update()

                    # Track training details
                    losses[epoch].append((self.loss.detach().item(), self.loss_1.detach().item(), self.loss_2.detach().item()))

            # Save model weights for subsequent inference
            torch.save(self, f"models/{name}/model_epoch_{epoch}.pt")
            torch.save(self, f"models/{name}/model_final.pt")
            # Print training details
            print(f"Epoch {epoch} completed in {time()-start} seconds")
            print(f"\tAverage Loss: ({sum([value[0] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])}, {sum([value[1] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])}, {sum([value[2] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])})")
        
        # Save final model
        torch.save(self, f"models/{name}/model_final.pt")
        return


"""
Generate predictions for a set of inputs.
    Inputs: Loaded model (model), location where results should be written to (name), data to predict on (dataloader)
    Outputs: None
"""
def get_predictions(model, name, dataloader):

    # Create directory to save visualizations in, if it does not yet exist
    if not os.path.exists(f"results/{name}/"):
        os.makedirs(f"results/{name}/")

    # To know where to store each result
    sample_no = 0
    # Generate predictions for the entire dataset
    for i, x in tqdm(enumerate(dataloader)):
        # Extract image from testing data (unsupervised, so this is all that is needed)
        image = (x["image"] / 255).to(device)
        # Only deal with complete batches for simplicity's sake
        if image.shape[0] != model.batch_size:
            break

        # Perform forward pass, computing model outputs without tracking training information
        with torch.no_grad():
            # Perform complete forward pass of the complete model
            _, _, log_masks, recon_comp_means = model(image)
            # Translate the data into a more useful format for visualization
            masks = torch.exp(log_masks)
            recon_comps = recon_comp_means

        # Generate a well-formatted summary image for each
        for j in tqdm(range(x["image"].shape[0])):
            # Summarize outputs as an image composed of a mosaic of images
            # Top row is the original image and reconstructed image 
            mosaic = [["original", "original", "original", "reconstruction", "reconstruction", "reconstruction"]]
            # Each subsequent row of images summarizes each of the K slots in the model
            for k in range(model.K):
                # First mask, then reconstructed-component, and then masked reconstructed-component
                mosaic.append([f"mask_{k}", f"mask_{k}", f"recon_comp_{k}", f"recon_comp_{k}", f"recon_comp_mask_{k}", f"recon_comp_mask_{k}"])
            # Generate mosaic of images
            _, axes = plt.subplot_mosaic(
                mosaic,
                height_ratios=[1.5] + [1] * model.K,       # Top images must be larger since there are only 2 (compared to 3)
                gridspec_kw={"wspace": 0, "hspace": 0},    # Ensure padding around images is tight
                constrained_layout=True,                   # Ensure padding around images is tight
                figsize=(
                    3 * 3 + 1,                             # Each image is 3*3, 3 columns, pad with 1
                    model.K * 3 + 1 + 5,                   # Each image is 3*3, K rows, pad with 1, add 5 for top row
                ),
            )
            
            # Display original image prominently
            axes["original"].imshow(x["image"][j].permute(1, 2, 0))
            axes["original"].set_title("Original", fontsize=25)
            # Display model's reconstructed image prominently (main result)
            axes["reconstruction"].imshow(torch.clamp(torch.sum(torch.mul(masks[:,j], recon_comps[:,j]), dim=0).permute(1, 2, 0), min=0, max=1))
            axes["reconstruction"].set_title("Reconstruction", fontsize=25)

            # Display labels for each column of the slot summary
            axes["mask_0"].set_title("Mask", fontsize=20)
            axes["recon_comp_0"].set_title("Recon.-Comp.", fontsize=20)
            axes["recon_comp_mask_0"].set_title("Masked Recon.-Comp.", fontsize=20)

            # Generate required summary plots for each slot
            for k in range(model.K):
                # Display label for each slot
                axes[f"mask_{k}"].set_ylabel(f"S{k}", rotation=0, va="center", labelpad=15, fontsize=20)
                # Display image for each important feature of plot
                axes[f"mask_{k}"].imshow(torch.clamp(masks[k][j].permute(1, 2, 0), min=0, max=1), cmap="gray")
                axes[f"recon_comp_{k}"].imshow(torch.clamp(recon_comps[k][j].permute(1, 2, 0), min=0, max=1))
                axes[f"recon_comp_mask_{k}"].imshow(torch.clamp(torch.mul(masks[k][j], recon_comps[k][j]).permute(1, 2, 0), min=0, max=1))

            # Set up figures to have proper formatting (no axis ticks/labels and square images)
            for key in axes.keys():
                axes[key].set_xticks([])
                axes[key].set_xticklabels([])
                axes[key].set_yticks([])
                axes[key].set_yticklabels([])
                axes[key].set_aspect("equal")
            
            # Save summary image
            plt.savefig(f"results/{name}/{sample_no}.png")
            plt.close()

            # One more sample has been analyzed
            sample_no += 1
    return


"""
Generate PyTorch DataLoader's to process a certain training dataset.
    Inputs: Names of datasets (names), whether to get training or testing data (split), batch size to train with (batch_size)
    Outputs: DataLoader's for the desired datasets (dataloaders)
"""
def get_data(names, split, batch_size):
    # Dataset overall size
    # Among datasets, Multi-dSprites is 64*64, Objects Room is 64*64, ...
    #     CLEVR is 240*320, Tetrominoes is 35*35, and ARC-AGI is 30*30 
    # Scale all data to 64*64 size, so 4096-dimensional
    # Size of tensors is (B, C, H, W) where B is batch size, C is number of channels, H and W are image size (64)

    # Load in datasets based on provided names
    datasets = []
    if ("CLEVR" in names) or ("All" in names):
        datasets.append(ClevrWithMasks(
            "datasets",
            # Crop exterior and blur down to 64*64
            transforms={
                "image": T.Compose([
                    T.CenterCrop(192),
                    T.Resize(64),
                ]),
            },
            split=split,
            download=False,
            convert=False,
        ))
    if ("MultiDSprites" in names) or ("All" in names):
        datasets.append(MultiDSprites(
            "datasets",
            version="colored_on_colored",
            split=split,
            download=False,
            convert=False,
        ))
    if ("ObjectsRoom" in names) or ("All" in names):
        datasets.append(ObjectsRoom(
            "datasets",
            split=split,
            download=False,
            convert=False,
        ))
    if ("Tetrominoes" in names) or ("All" in names):
        datasets.append(Tetrominoes(
            "datasets",
            # Expand up to 64*64
            transforms={
                "image": T.Compose([
                    T.CenterCrop(32),
                    T.Resize(64, interpolation=T.InterpolationMode.NEAREST),
                ]),
            },
            split=split,
            download=False,
            convert=False,
        ))
    if ("ARCAGI" in names) or ("All" in names):
        datasets.append(ARCAGI(
            "datasets",
            version="zoomed",
            split=split,
        ))

    # Transform into DataLoader objects that can be directly used in training
    dataloaders = [DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    ) for dataset in datasets]

    return dataloaders


"""
Manage input and output for running the program.
"""
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "--n", nargs=1, required=True, type=str, help="Name of model to save or load.")
    parser.add_argument("--mode", "--m", nargs=1, required=True, type=str, choices=["Train", "Test"], help="Whether to train a new model or test an existing one.")
    parser.add_argument("--datasets", "--d", nargs="+", required=True, type=str, choices=["All", "CLEVR", "MultiDSprites", "ObjectsRoom", "Tetrominoes", "ARCAGI"], help="Datasets to use for training/testing.")
    parser.add_argument("--num_slots", "--s", nargs=1, required=False, default=[5], type=int, help="Number of attention slots in the model.")
    parser.add_argument("--batch-size", "--b", nargs=1, required=True, type=int, help="Number of images to be processed in a single batch.")
    parser.add_argument("--epochs", "--e", nargs=1, required=False, default=[1000], type=int, help="Number of epochs to complete during training.")
    parser.add_argument("--learning-rate", "--l", nargs=1, required=False, default=[0.0001], type=float, help="Learning rate to be used during model training.")
    options = parser.parse_args()
    return options


def main():
    # Model setup and hyperparameters
    options = parse_options()
    name = options.name[0]
    mode = options.mode[0]
    datasets = options.datasets
    num_slots = options.num_slots[0]
    batch_size = options.batch_size[0]
    epochs = options.epochs[0]
    learning_rate = options.learning_rate[0]

    # Set up datasets
    print(f"{mode}ing {name} on {', '.join(datasets)}.")
    dataloaders = get_data(datasets, mode, batch_size)

    # Train a new model
    if mode == "Train":
        print(f"\t{mode}ing {sum(p.numel() for p in model.parameters() if p.requires_grad)}-parameter model with {num_slots} slots in batches of {batch_size} with a learning rate of {learning_rate} for {epochs} epochs.")
        model = MONet_NoRMask(num_slots, batch_size, learning_rate).to(device)
        model.learn(dataloaders, epochs, name)

    # Test an existing model
    elif mode == "Test":
        model = MONet_NoRMask(num_slots, batch_size, 0)
        model.load_state_dict(
            torch.load(
                f"models/{name}.pt",
                weights_only=False,
                map_location=torch.device(device),
            )
        )
        model.eval()
        get_predictions(model, name, dataloaders[0])

    # Finalize run
    print(f"{mode}ing {name} on {', '.join(datasets)} complete.")
    return


if __name__ == "__main__":
    main()
