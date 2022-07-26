import zipfile
from pathlib import Path
from typing import Tuple, Union

import gdown
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from not_ae.utils.general import DATA_DIR, REGISTRY
from not_ae.utils.transform import NormalizeInverse


N_CIFAR_CLASSES = 10


def download_celeba():
    data_root = Path(DATA_DIR, "celeba")
    data_root.mkdir(exist_ok=True)

    # URL for the CelebA dataset
    url = "https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH"

    download_path = Path(data_root, "img_align_celeba.zip")
    gdown.download(url, download_path.as_posix(), quiet=False)

    url = "https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk"
    download_path = Path(data_root, "list_eval_partition.txt")
    gdown.download(url, download_path.as_posix(), quiet=False)

    with zipfile.ZipFile(download_path, "r") as ziphandler:
        ziphandler.extractall(data_root)


@REGISTRY.dataset.register()
class CelebADataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path] = DATA_DIR,
        split: str = "train",
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        img_size: int = 64,
        # transform: Optional[Any] = None,
    ):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        self.mean, self.std = mean, std
        self.split = split
        # Read names of images in the root directory
        img_folder = Path(root_dir, "celeba", "img_align_celeba")
        if not img_folder.exists():
            download_celeba()

        list_partition = pd.read_csv(
            Path(root_dir, "celeba", "list_eval_partition.txt"), sep=" ", header=None
        )

        self.image_names = {}
        for split_, value in zip(["train", "val", "test"], range(3)):
            self.image_names[split_] = (
                list_partition[list_partition.iloc[:, 1] == value].iloc[:, 0].tolist()
            )

        self.norm_transform = T.Normalize(
            mean=mean,
            std=std,
        )
        self.transform = T.Compose(
            [
                T.CenterCrop(178),  # Because each image is size (178, 218) spatially.
                T.Resize(img_size),
                T.ToTensor(),
                self.norm_transform,
            ]
        )
        self.inverse_transform = NormalizeInverse(mean, std)
        self.img_folder = img_folder

    def __len__(self):
        return len(self.image_names[self.split])

    def __getitem__(self, idx):
        # Get the path to the image
        img_path = Path(self.img_folder, self.image_names[self.split][idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img
