from typing import Tuple

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from not_ae.utils.general import REGISTRY
from not_ae.utils.transform import NormalizeInverse


@REGISTRY.dataset.register()
class ArtBench10(CIFAR10):

    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "b116ffdc5e07e162f119149c2ad7403f"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )
        self.inverse_transform = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index: int):
        out = super().__getitem__(index)
        if isinstance(out, Tuple):
            return out[0]
        else:
            return out
