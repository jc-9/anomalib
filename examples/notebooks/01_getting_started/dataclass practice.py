import torch
from anomalib.data import ImageBatch

img_batch_path = '/Users/justinclay/PycharmProjects/Pytorch Tutorials Sep 25 2025/anomalib/examples/notebooks/01_getting_started/datasets/MVTecAD/bottle/train/good'

batch = ImageBatch(
    image=torch.rand((8, 3, 224, 224)),
    gt_label=[0, ] * 8,
    gt_mask=torch.zeros((8, 224, 224)),
)