from PIL import Image
from torchvision.transforms import Compose
from data.transform import Open, Normalize, Tensor, Resize, ColorizeLabels, custom_collate
from data.cityscapes import Cityscapes
from pathlib import Path
from torch.utils.data import DataLoader, Subset


def prepare_data(root_path, subset='val', num_images=None, image_size=(1024, 2048)):
    transforms = Compose([
        Open(),
        Resize((image_size[1], image_size[0])),
        Normalize(scale=255, mean=Cityscapes.mean, std=Cityscapes.std),
        Tensor()
    ])

    dataset = Cityscapes(Path(root_path), transforms=transforms, subset=subset)

    if num_images and num_images < len(dataset):
        indices = list(range(num_images))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)

    return loader


def visualize_and_save_segmentation_result(predictions, output_path):
    color_info = Cityscapes.color_info
    to_color = ColorizeLabels(color_info)

    colored_pred = to_color(predictions)
    colored_pred_img = Image.fromarray(colored_pred)
    colored_pred_img.save(output_path)
