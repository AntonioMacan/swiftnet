import numpy as np
from PIL import Image
from torchvision.transforms import Compose
from data.transform import Open, Normalize, Tensor, ColorizeLabels
from data.cityscapes import Cityscapes


def load_and_preprocess_image(image_path):
    scale = 255
    mean = Cityscapes.mean
    std = Cityscapes.std

    transforms = Compose([
        Open(),
        Normalize(scale=scale, mean=mean, std=std),
        Tensor()
    ])
    
    # Load image
    sample = {"image": image_path}
    sample = transforms(sample)
    
    # Add batch dimension
    image_tensor = sample["image"].unsqueeze(0)   # (1, 3, H, W)

    # Convert to numpy float32
    image_np = image_tensor.numpy().astype(np.float32)
    return image_np


def visualize_and_save_segmentation_result(predictions, output_path):
    color_info = Cityscapes.color_info
    to_color = ColorizeLabels(color_info)

    colored_pred = to_color(predictions)
    colored_pred_img = Image.fromarray(colored_pred)
    colored_pred_img.save(output_path)
    print(f"Saved segmentation result to: {output_path}")
