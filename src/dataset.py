import torch
from torch.utils.data import Dataset
import os
import random
from torchvision.io import read_image
from torchvision.transforms import v2


class CatDataset(Dataset):
    def __init__(self, dataset_path, part, seed, val_part, limit=None):
        super().__init__()
        random.seed(seed)
        assert part in ['train', 'val'], "Part should be 'train', 'val'"
        
        self.dataset_path = dataset_path
        self.part = part
        self.val_part = val_part

        self.index = self._make_index()
        if limit is not None:
            self.index = self.index[:limit]

    def _make_index(self,):
        paths = sorted(os.listdir(self.dataset_path))
        paths = [os.path.join(self.dataset_path, x) for x in paths]
        random.shuffle(paths)
        
        train_size = int(len(paths) * (1 - self.val_part))
        if self.part == 'train':
            paths = paths[:train_size]
        else:
            paths = paths[train_size:]
        return paths
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, ind):
        path = self.index[ind]
        img = read_image(path)
        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transforms(img) 
        return img

# def collate_fn(dataset_items):
#     audios = [elem["audio"][0] for elem in dataset_items]
#     labels = [elem["label"] for elem in dataset_items]
#     audios = pad_sequence(audios, batch_first=True).unsqueeze(1)
#     return audios, torch.tensor(labels)