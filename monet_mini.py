##########################################
#####           mini-MONet           #####
##########################################

# pytorch
import torch
import torch.nn as nn

# dataset batching and image transforms
from torch.utils.data import DataLoader
from torchvision import transforms as T

# training and testing tools
import os
import argparse
from tqdm import tqdm
from time import time
from multi_object_datasets_torch import ClevrWithMasks, MultiDSprites, ObjectsRoom, Tetrominoes
import matplotlib.pyplot as plt

# get GPU information
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device being used: {device}.")
if device == "cuda":
    print(f"Number of GPUs available: {torch.cuda.device_count()}.")
    torch.backends.cudnn.benchmark = True


# attention network
class AttentionNetwork(nn.Module):

    def __init__(self, batchsize):
        super().__init__()
        self.batchsize = batchsize

        # input is RGB image + scope
        # input: (batch size, number of channels, image height, image width)
        # (B, C, H, W) = (64, 4, 32, 32)

        # down block 1
        self.downblock1 = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 8, affine = True),
            nn.ReLU()) # (B, C, H, W) = (64, 8, 32, 32)
        self.downsample1 = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = True)
        # (B, C, H, W) = (64, 8, 16, 16)

        # down block 2
        self.downblock2 = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 16, affine = True),
            nn.ReLU()) # (B, C, H, W) = (64, 16, 16, 16)
        self.downsample2 = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = True)
        # (B, C, H, W) = (64, 16, 8, 8)

        # down block 3
        self.downblock3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 32, affine = True),
            nn.ReLU()) # (B, C, H, W) = (64, 32, 8, 8)
        self.downsample3 = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0, ceil_mode = True)
        # (B, C, H, W) = (64, 32, 4, 4)

        # down block 4
        self.downblock4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 64, affine = True),
            nn.ReLU()) # (B, C, H, W) = (64, 64, 4, 4)
        
        # 3-layer MLP
        self.MLP = nn.Sequential(
            nn.Flatten(), # flatten to (64, 64 x 4 x 4 = 1024, 1, 1)
            nn.Linear(in_features = 1024, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 1024),
            nn.ReLU()) # reshape to (64, 64, 4, 4)
        
        # up blocks
        self.upblock1 = nn.Sequential(
            # concatenate with (64, 64, 4, 4) to get (64, 128, 4, 4)
            nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 32, affine = True),
            nn.ReLU(), # (B, C, H, W) = (64, 32, 4, 4)
            nn.Upsample(scale_factor = 2, mode = "nearest")) # (B, C, H, W) = (64, 32, 8, 8)
        self.upblock2 = nn.Sequential(
            # concatenate with (64, 32, 8, 8) to get (64, 64, 8, 8)
            nn.Conv2d(in_channels = 64, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 16, affine = True),
            nn.ReLU(), # (B, C, H, W) = (64, 16, 8, 8)
            nn.Upsample(scale_factor = 2, mode = "nearest")) # (B, C, H, W) = (64, 16, 16, 16)
        self.upblock3 = nn.Sequential(
            # concatenate with (64, 16, 16, 16) to get (64, 32, 16, 16)
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 8, affine = True),
            nn.ReLU(), # (B, C, H, W) = (64, 8, 16, 16)
            nn.Upsample(scale_factor = 2, mode = "nearest")) # (B, C, H, W) = (64, 8, 32, 32)
        self.upblock4 = nn.Sequential(
            # concatenate with (64, 8, 32, 32) to get (64, 16, 32, 32)
            nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(num_features = 4, affine = True),
            nn.ReLU()) # (B, C, H, W) = (64, 4, 32, 32)

        # final output layer produces the attention mask at recurrent step k
        self.finalattention = nn.Conv2d(in_channels = 4, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # (B, C, H, W) = (64, 1, 32, 32)

    def forward(self, x, log_scope_k):

        # RGB image + scope
        x0 = torch.concat((x, log_scope_k), dim = 1)

        # down the U-net
        x1 = self.downblock1(x0) # first skip tensor
        x2 = self.downblock2(self.downsample1(x1)) # second skip tensor
        x3 = self.downblock3(self.downsample2(x2)) # third skip tensor
        x4 = self.downblock4(self.downsample3(x3)) # fourth skip tensor

        # through the 3-layer MLP
        y = self.MLP(x4).reshape(self.batchsize, 64, 4, 4)

        # up the U-net
        y = self.upblock1(torch.concat((y, x4), dim = 1)) # concat first skip tensor
        y = self.upblock2(torch.concat((y, x3), dim = 1)) # concat second skip tensor
        y = self.upblock3(torch.concat((y, x2), dim = 1)) # concat third skip tensor
        y = self.upblock4(torch.concat((y, x1), dim = 1)) # concat fourth skip tensor

        # final layer
        alpha_k = self.finalattention(y)

        # compute scope and mask
        log_alpha_k = nn.LogSigmoid()(alpha_k)
        log_scope_kplus1 = log_scope_k + log_alpha_k - alpha_k
        log_mask_k = log_scope_k + log_alpha_k

        # output scope and mask
        return log_scope_kplus1, log_mask_k


# component variational autoencoder
class ComponentVAE(nn.Module):

    def __init__(self, batchsize):
        super().__init__()
        self.batchsize = batchsize

        # input is RGB image + attention mask at recurrent step k
        # input: (batch size, number of channels, image height, image width)
        # (B, C, H, W) = (64, 4, 32, 32)

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.ReLU(), # (B, C, H, W) = (64, 16, 16, 16)
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.ReLU(), # (B, C, H, W) = (64, 32, 8, 8)
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.ReLU(), # (B, C, H, W) = (64, 32, 4, 4)
            nn.Flatten(), # flatten to (64, 32 x 4 x 4 = 512, 1, 1)
            nn.Linear(in_features = 512, out_features = 128),
            nn.Linear(in_features = 128, out_features = 32)) # 16-dimensional latent representation

        # sample latent distribution: (B, C, H, W) = (64, 16, 1, 1)
        # spatial broadcast/tiling: (B, C, H, W) = (64, 16, 38, 38)
        # concat coordinate channels: (B, C, H, W) = (64, 18, 38, 38)

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels = 18, out_channels = 32, kernel_size = 3, stride = 1, padding = 0, bias = True),
            nn.ReLU(), # (B, C, H, W) = (64, 32, 36, 36)
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0, bias = True),
            nn.ReLU(), # (B, C, H, W) = (64, 32, 34, 34)
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 0, bias = True),
            nn.ReLU(), # (B, C, H, W) = (64, 16, 32, 32)
            nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size = 1, stride = 1, padding = 0, bias = True))
            # final output layer produces reconstructed component and mask
            # (B, C, H, W) = (64, 4, 32, 32)

    def forward(self, x, log_mask_k):

        # RGB image + attention mask at recurrent step k
        x0 = torch.concat((x, log_mask_k), dim = 1)

        # encode
        latent = self.encoder(x0)

        # extract latent parameters mu and log sigma
        mu = torch.split(latent, 16, dim = 1)[0]
        log_sigma = torch.split(latent, 16, dim = 1)[1]

        # sample from the distribution
        z = mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)

        # broadcast the latent vector across space
        z = z.reshape((self.batchsize, 16, 1, 1)).repeat((1, 1, 38, 38))

        # coordinate channels for each spacial dimension
        dim1 = torch.linspace(-1, 1, 38, device = device)
        dim2 = torch.linspace(-1, 1, 38, device = device)
        dim1, dim2 = torch.meshgrid(dim1, dim2, indexing = "ij")
        dim1 = dim1.reshape((1, 1, 38, 38)).repeat((self.batchsize, 1, 1, 1))
        dim2 = dim2.reshape((1, 1, 38, 38)).repeat((self.batchsize, 1, 1, 1))

        # z + coordinate channels
        z = torch.concat((z, dim1, dim2), dim = 1)

        # decode
        reconstructed = self.decoder(z)

        # extract reconstructed image and mask
        x_hat_mu = torch.split(reconstructed, [3, 1], dim = 1)[0]
        mask_hat_logits = torch.split(reconstructed, [3, 1], dim = 1)[1]

        # output latent parameters and reconstructed component and mask
        return mu, log_sigma, x_hat_mu, mask_hat_logits


# mini monet
class MiniMONet(nn.Module):

    def __init__(self, slots, batchsize, learningrate):
        super().__init__()

        # total number of recurrent steps and total number of attention masks
        self.K = slots

        # initialize networks
        self.batchsize = batchsize
        self.attn = AttentionNetwork(self.batchsize)
        self.vae = ComponentVAE(self.batchsize)

        # tuning parameters for the terms of the loss function
        self.alpha = 1
        self.beta = 0.5
        self.gamma = 0.5

        # output component distribution is an independent pixel-wise Gaussian with ﬁxed scales
        self.sigma_bg = 0.09 # background
        self.sigma_fg = 0.11 # foreground

        # optimize learning using RMSProp
        self.learningrate = learningrate
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr = self.learningrate)

    def forward(self, x):

        # attention network output
        log_scope = torch.zeros((self.K, self.batchsize, 1, 32, 32), device = device)
        log_scope[0] = torch.log(torch.ones((self.batchsize, 1, 32, 32), device = device))
        log_mask = torch.zeros((self.K, self.batchsize, 1, 32, 32), device = device)

        # VAE latent distribution parameters
        mu = torch.zeros((self.K, self.batchsize, 16), device = device)
        log_sigma = torch.zeros((self.K, self.batchsize, 16), device = device)
        
        # VAE output
        x_hat_mu = torch.zeros((self.K, self.batchsize, 3, 32, 32), device = device)
        mask_hat_logits = torch.zeros((self.K, self.batchsize, 1, 32, 32), device = device)

        # run K - 1 recurrent steps and send attention masks to VAE
        for k in range(self.K - 1):
            log_scope[k + 1], log_mask[k] = self.attn(x, log_scope[k])
            mu[k], log_sigma[k], x_hat_mu[k], mask_hat_logits[k] = self.vae(x, log_mask[k])
        
        # run Kth recurrent step and send attention mask to VAE
        log_mask[self.K - 1] = log_scope[self.K - 1]
        mu[self.K - 1], log_sigma[self.K - 1], x_hat_mu[self.K - 1], mask_hat_logits[self.K - 1] = self.vae(x, log_mask[self.K - 1])

        # output attention mask and latent parameters and reconstructed component and mask
        return mu, log_sigma, log_mask, x_hat_mu, mask_hat_logits

    def learn(self, dataloaders, epochs, name):

        # create directory to save model in
        if not os.path.exists(f"models/{name}/"):
            os.makedirs(f"models/{name}/")

        # optimize training
        scaler = torch.amp.GradScaler("cuda")

        # here we go
        losses = []
        for epoch in tqdm(range(epochs)):

            # time the epoch
            start = time()

            # iterate over each dataset
            losses.append([])
            for dataloader in dataloaders:

                # iterate over each batch
                for i, x in enumerate(dataloader):

                    # normalize pixel values
                    image = (x["image"] / 255).to(device)

                    # skip batches that are not full
                    if image.shape[0] != self.batchsize: break

                    # compute loss
                    self.optimizer.zero_grad(set_to_none = True)
                    with torch.autocast(device):

                        # forward pass for a batch
                        log_mask, mu, log_sigma, x_hat_mu, mask_hat_logits = self(image)

                        # output component distribution is an independent pixel-wise Gaussian with ﬁxed scales
                        x_hat_sigma = torch.Tensor([self.sigma_bg if k == 0 else self.sigma_fg for k in range(self.K)])

                        # first loss - VAE decoder negative log likelihood
                        # weighted with hyperparameter alpha = 1
                        loss1 = 0
                        for j in range(0, self.K):
                            temp = torch.exp(log_mask[j] - torch.log(x_hat_sigma[j]) - 0.5 * (image - x_hat_mu[j]).pow(2) / x_hat_sigma[j].pow(2))
                            loss1 = loss1 + temp
                        loss1 = torch.sum(-torch.log(loss1))
                        loss1 = self.alpha * loss1 / self.batchsize

                        # second loss - VAE KL divergence of latent posterior factored across slots and latent prior
                        # weighted with hyperparameter beta which encourages learning of disentangled latent representations
                        loss2 = 0
                        for j in range(0, self.K):
                            temp = torch.sum(torch.exp(log_sigma[j]).pow(2) + mu[j].pow(2) - 2 * log_sigma[j] - 1) / 2 # from lecture notes pg. 113/114
                            loss2 = loss2 + temp
                        loss2 = self.beta * loss2 / self.batchsize

                        # third loss - KL divergence of attention mask distribution and VAE decoded mask distribution
                        # weighted with hyperparameter gamma which modulates how closely the VAE must model the attention mask distribution
                        loss3 = nn.KLDivLoss(reduction = "sum", log_target = True)(nn.LogSoftmax(dim = 0)(mask_hat_logits), log_mask)
                        loss3 = self.gamma * loss3 / self.batchsize

                        # total loss
                        loss = loss1 + loss2 + loss3

                    # backward pass and parameter update
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
                    scaler.update()

                    # training history
                    losses[epoch].append((loss.detach().item(), loss1.detach().item(), loss2.detach().item(), loss3.detach().item()))

            # save model weights for subsequent inference
            torch.save(self, f"models/{name}/model_final.pt")
            torch.save(self.state_dict(), f"models/{name}/model_final_state.pt")
            if (epoch + 1) % (epochs // 25) == 0:
                torch.save(self, f"models/{name}/model_epoch_{epoch}.pt")
                torch.save(self.state_dict(), f"models/{name}/model_epoch_{epoch}_state.pt")

            # print training details
            print(f"Epoch {epoch} completed in {time() - start} seconds")
            print(f"\tAverage Loss: ({sum([value[0] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])}, "
                  f"{sum([value[1] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])}, "
                  f"{sum([value[2] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])}, "
                  f"{sum([value[3] / len(dataloaders) / len(dataloaders[0]) for value in losses[-1]])})")

        # save final model
        torch.save(self, f"models/{name}/model_final.pt")
        torch.save(self.state_dict(), f"models/{name}/model_final_state.pt")

        # done studying
        return


def get_predictions(model, name, dataloader):

    # Create directory to save visualizations in, if it does not yet exist
    if not os.path.exists(f"results/{name}"):
        os.makedirs(f"results/{name}")

    # To know where to store each result
    sample_no = 0
    # Generate predictions for the entire dataset
    for i, x in tqdm(enumerate(dataloader)):
        # Extract image from testing data (unsupervised, so this is all that is needed)
        image = (x["image"] / 255).to(device)
        # Only deal with complete batches for simplicity's sake
        if image.shape[0] != model.batchsize:
            break

        # Perform forward pass, computing model outputs without tracking training information
        with torch.no_grad():
            # Perform complete forward pass of the complete model
            _, _, log_masks, recon_comp_means, recon_mask_logits = model(image)
            # Translate the data into a more useful format for visualization
            masks = torch.exp(log_masks)
            recon_comps = recon_comp_means
            recon_masks = nn.Softmax(dim=0)(recon_mask_logits)

        # Generate a well-formatted summary image for each
        for j in tqdm(range(x["image"].shape[0])):
            # Summarize outputs as an image composed of a mosaic of images
            # Top row is the original image and reconstructed image 
            mosaic = [["original", "original", "original", "original", "original", "reconstruction", "reconstruction", "reconstruction", "reconstruction", "reconstruction"]]
            # Each subsequent row of images summarizes each of the K slots in the model
            for k in range(model.K):
                # First mask, then reconstructed-mask, then reconstructed-component, then masked reconstructed-component, and finally reconstructed-masked reconstructed-component
                mosaic.append([f"mask_{k}", f"mask_{k}", f"recon_mask_{k}", f"recon_mask_{k}", f"recon_comp_{k}", f"recon_comp_{k}", f"recon_comp_mask_{k}", f"recon_comp_mask_{k}", f"recon_comp_recon_mask_{k}", f"recon_comp_recon_mask_{k}"])
            # Generate mosaic of images
            _, axes = plt.subplot_mosaic(
                mosaic,
                height_ratios=[2.5] + [1] * model.K,       # Top images must be larger since there are only 2 (compared to 5)
                gridspec_kw={"wspace": 0, "hspace": 0},    # Ensure padding around images is tight
                constrained_layout=True,                   # Ensure padding around images is tight
                figsize=(
                    5 * 3 + 1,                             # Each image is 3*3, 5 columns, pad with 1
                    model.K * 3 + 1 + 8,                   # Each image is 3*3, K rows, pad with 1, add 8 for top row
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
            axes["recon_mask_0"].set_title("Recon.-Mask", fontsize=20)
            axes["recon_comp_0"].set_title("Recon.-Comp.", fontsize=20)
            axes["recon_comp_mask_0"].set_title("Masked Recon.-Comp.", fontsize=20)
            axes["recon_comp_recon_mask_0"].set_title("R.-Masked R.-Comp.", fontsize=20)

            # Generate required summary plots for each slot
            for k in range(model.K):
                # Display label for each slot
                axes[f"mask_{k}"].set_ylabel(f"S{k}", rotation=0, va="center", labelpad=15, fontsize=20)
                # Display image for each important feature of plot
                axes[f"mask_{k}"].imshow(torch.clamp(masks[k][j].permute(1, 2, 0), min=0, max=1), cmap="gray")
                axes[f"recon_mask_{k}"].imshow(torch.clamp(recon_masks[k][j].permute(1, 2, 0), min=0, max=1), cmap="gray")
                axes[f"recon_comp_{k}"].imshow(torch.clamp(recon_comps[k][j].permute(1, 2, 0), min=0, max=1))
                axes[f"recon_comp_mask_{k}"].imshow(torch.clamp(torch.mul(masks[k][j], recon_comps[k][j]).permute(1, 2, 0), min=0, max=1))
                axes[f"recon_comp_recon_mask_{k}"].imshow(torch.clamp(torch.mul(recon_masks[k][j], recon_comps[k][j]).permute(1, 2, 0), min=0, max=1))

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
    datasets_names = options.datasets
    num_slots = options.num_slots[0]
    batch_size = options.batch_size[0]
    epochs = options.epochs[0]
    learning_rate = options.learning_rate[0]

    # Set up datasets
    print(f"{mode}ing {name} on {', '.join(datasets_names)}.")
    datasets = []
    if "MultiDSprites" in datasets_names:
        datasets.append(MultiDSprites("datasets", version = "colored_on_colored", split = "Train",
            transforms = {"image": T.Resize(32, T.InterpolationMode.NEAREST)},
            download = False, convert = False))
        datasets.append(MultiDSprites("datasets", version = "colored_on_grayscale", split = "Train",
            transforms = {"image": T.Resize(32, T.InterpolationMode.NEAREST)},
            download = False, convert = False))
    elif "ObjectsRoom" in datasets_names:
        datasets.append(ObjectsRoom("datasets", split = "Train",
            transforms = {"image": T.Resize(32, T.InterpolationMode.NEAREST)},
            download = False, convert = False))
    dataloaders = [DataLoader(dataset, batch_size = batch_size, num_workers = 4, shuffle = True, pin_memory = True) for dataset in datasets]

    # Train a new model
    if mode == "Train":
        print(f"\t{mode}ing with {num_slots} slots in batches of {batch_size} with a learning rate of {learning_rate} for {epochs} epochs.")
        mini_MONet = MiniMONet(num_slots, batch_size, learning_rate).to(device)
        mini_MONet.learn(dataloaders, epochs, name)

    # Test an existing model
    elif mode == "Test":
        mini_MONet = MiniMONet(num_slots, batch_size, 0)
        mini_MONet.load_state_dict(
            torch.load(
                f"models/{name}.pt",
                weights_only=False,
                map_location=torch.device(device),
            )
        )
        mini_MONet.eval()
        get_predictions(mini_MONet, name, dataloaders[0])

    # Finalize run
    print(f"{mode}ing {name} on {', '.join(datasets_names)} complete.")
    return

if __name__ == "__main__":
    main()
