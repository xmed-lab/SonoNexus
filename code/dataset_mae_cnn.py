import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import torch

def get_all_image_paths(root_folder):
    image_paths = []
    # Walk through all directories and subdirectories
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if the file is an image by extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Get the absolute path and add it to the list
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def get_transform(img_size):
    return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.16, 1.0), ratio=(4/5, 5/4), interpolation=3),  
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])


class Mask_Dataset(Dataset):
    def __init__(self, img_dir='/public/LLM/Videos', transform=None, upper_ratio=0.8, lower_ratio=0.5, block_size=8, img_size=224):

        print("lower ratio {}; upper ratio {}; block size {}; img size {} of Datsets".format(lower_ratio, upper_ratio, block_size, img_size))

        self.img_dir = img_dir
        self.transform = transform
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.block_size = block_size
        self.img_labels = get_all_image_paths(img_dir)
        
        self.num_blocks_vertical = img_size // block_size
        self.num_blocks_horizontal = img_size // block_size
        self.num_blocks = self.num_blocks_vertical * self.num_blocks_horizontal
        self.idx_blocks = range(self.num_blocks)
        
        nums = int((upper_ratio - lower_ratio + 1e-8) // 0.05 + 1)
        self.decimals = [lower_ratio + 0.05*i for i in range(nums)]
        
    def generate_mask(self):
        curr_ratio = random.uniform(self.lower_ratio, self.upper_ratio) #random.choice(self.decimals)
        num_mask_blocks = int(self.num_blocks * curr_ratio)

        mask_blocks_indices = random.sample(self.idx_blocks, num_mask_blocks)
        mask = np.zeros((self.num_blocks_vertical, self.num_blocks_horizontal), dtype=bool)
        np.put(mask, mask_blocks_indices, True)
        mask_expanded = np.repeat(mask, self.block_size, axis=0)
        mask_expanded = np.repeat(mask_expanded, self.block_size, axis=1)
        
        mask = torch.FloatTensor(mask_expanded).unsqueeze(0)
        return mask, curr_ratio
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.img_labels[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        mask, _ = self.generate_mask()
        
        return image, mask, _


def get_data(data_root="/public/LLM/Videos", batch_size=2, num_workers=4, img_size=224, patch_size=8):
    trans = get_transform(img_size=img_size)
    set_ = Mask_Dataset(data_root, trans, img_size=img_size, block_size=patch_size)

    print("The total number of images if {}".format(len(set_)))

    shuffle = True
    return DataLoader(dataset=set_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    data_root = "/public/LLM/Videos"
    batch_size = 32
    data_loader = get_data(data_root, batch_size, num_workers=1, img_size=224, patch_size=8)
    for sample in data_loader:
        images, masks, ratios = sample
        print(images.size(), masks.size(), images.min(), images.max())
        break

    N = 10
    imgs = []
    mks = []
    for i in range(N):
        img = (images[i].permute(1,2,0).detach().cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        mask = masks[i].permute(1,2,0).detach().cpu().numpy()
        #print(mask.shape)
        img_mask = (img * (1-mask)).astype(np.uint8)
        imgs.append(img)
        mks.append(img_mask)

    fig, axs = plt.subplots(2, N, figsize=(25, 7))

# 展示原始图像（第一排）
    for i, ax in enumerate(axs[0]):
        ax.imshow(imgs[i])
        ax.axis('off')  # 隐藏坐标轴
        ax.set_title(f"Original {i+1}")

# 展示处理后的图像（第二排）
    for i, ax in enumerate(axs[1]):
        ax.imshow(mks[i])
        ax.axis('off')  # 隐藏坐标轴
        ax.set_title("ratio {}".format(round(ratios[i].item(), 3)))

    plt.savefig("./test.png")