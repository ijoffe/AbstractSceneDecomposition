# Isaac Joffe and Benjamin Colussi, 2025


# Required imports to utilize the custom ARC-AGI dataset
from torchvision.datasets import VisionDataset
import torch
# Required imports to build the custom ARC-AGI dataset
from torchvision import transforms as T
from PIL import ImageColor
import json
import os
from tqdm import tqdm


# Note: Unused, not optimized enough
# """
# Class to train on multiple datasets at once.
# """
# class MONetDataset(VisionDataset):
#     def __init__(self, datasets):
#         self.data = []
#         for i in range(len(datasets)):
#             for j in range(len(datasets[i])):
#                 self.data.append({"image": datasets[i][j]["image"]})
#         return

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)


"""
PyTorch-usable wrapper for the custom ARC-AGI dataset.
"""
class ARCAGI(VisionDataset):
    def __init__(self, root, version="zoomed", split="Train"):
        super().__init__(root)
        assert version in ["padded", "centred", "zoomed"], "Invalid dataset requested"
        assert split in ["Train", "Test"], "Invalid dataset requested"
        path = f"{root}/arc-agi-datasets/arc_agi_data/arc_agi_data_{version}_{split.lower()}.pt"
        self.data = torch.load(path, weights_only=False)
        return    

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


"""
Function to generate custom ARC-AGI datasets.
"""
def make_dataset(root, src_dir, version, split):
    # Build mapping of ARC-AGI colours onto usable RGB colours
    colors  = {
        0: "#000000",
        1: "#0074D9",
        2: "#FF4136",
        3: "#2ECC40",
        4: "#FFDC00",
        5: "#AAAAAA",
        6: "#F012BE",
        7: "#FF851B",
        8: "#7FDBFF",
        9: "#870C25",
    }
    for key in colors.keys():
        colors[key] = list(ImageColor.getcolor(colors[key], "RGB"))

    dataset = []
    # Build up dataset by considering each ARC-AGI task
    for task in tqdm(os.listdir(src_dir)):
        assert version in ["padded", "centred", "zoomed"], "Invalid dataset requested"
        assert split in ["train", "test"], "Invalid dataset requested"

        # Read in data for each task, simple JSON format
        with open(src_dir + task) as f:
            data = json.load(f)

        # Store all grids for this task, regardless of what role they play in the challenge
        grids = []
        for sample in data["train"]:
            grids.append(sample["input"])
            grids.append(sample["output"])
        for sample in data["test"]:
            grids.append(sample["input"])
            grids.append(sample["output"])

        # Process each grid to be usable by the model
        for grid in grids:
            # Make initially empty grid tensor with proper dimensions and datatype
            grid_tensor = torch.zeros((3, len(grid), len(grid[0])), dtype=torch.uint8)
            # Fill in grid tensor, mapping each ARC-AGI pixel onto its RGB representation
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    grid_tensor[:, i, j] = torch.Tensor(colors[grid[i][j]])

            if version == "padded":
                # Expand image to 64*64, with content in top-left corner
                grid_tensor = T.Pad((0, 0, 64 - len(grid[0]), 64 - len(grid)))(grid_tensor)
            elif version == "centred":
                # Expand image to 64*64, with content centred
                grid_tensor = T.CenterCrop(64)(grid_tensor)
            elif version == "zoomed":
                # Expand image to 64*64, with content blown up
                grid_tensor = T.Resize(64, interpolation=T.InterpolationMode.NEAREST)(T.CenterCrop(max(len(grid), len(grid[0])))(grid_tensor))
            assert grid_tensor.shape[0] == 3 and grid_tensor.shape[1] == 64 and grid_tensor.shape[2] == 64

            # Add sample to dataset, tracking which ARC-AGI task it corresponds to
            dataset.append({
                "image": grid_tensor,
                "name": task.split(".")[0],
            })

    # Save dataset built
    path = f"{root}/arc-agi-datasets/arc_agi_data/arc_agi_data_{version}_{split}.pt"
    torch.save(dataset, path)
    return


def main():
    # Construct each dataset
    root = "datasets"
    versions = ["padded", "centred", "zoomed"]

    # Construct each training dataset
    src_dir = "datasets/arc-agi-datasets/data/training/"
    split = "train"
    for version in versions:
        make_dataset(root, src_dir, version, split)

    # Construct each testing dataset
    src_dir = "datasets/arc-agi-datasets/data/evaluation/"
    split = "test"
    for version in versions:
        make_dataset(root, src_dir, version, split)
    return


if __name__ == "__main__":
    main()
