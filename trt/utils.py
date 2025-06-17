import numpy as np
from PIL import Image
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

from data.transform import Open, Normalize, Tensor, Resize, ColorizeLabels
from data.cityscapes import Cityscapes
from pathlib import Path


def custom_collate(batch, del_orig_labels=False):
    keys = ['target_size', 'target_size_feats', 'alphas', 'target_level']
    values = {}
    for k in keys:
        if k in batch[0]:
            values[k] = batch[0][k]
    for b in batch:
        if del_orig_labels: 
            del b['original_labels']
        for k in values.keys():
            del b[k]
        if 'mux_indices' in b:
            b['mux_indices'] = b['mux_indices'].view(-1)
    batch = default_collate(batch)
    for k, v in values.items():
        batch[k] = v
    return batch


def prepare_data(root_path, subset='val', num_images=None, batch_size=1, image_size=(1024, 2048)):
    transforms = Compose([
        Open(),
        Resize((image_size[1], image_size[0])),  # Note: SwiftNet expects (width, height)
        Normalize(scale=255, mean=Cityscapes.mean, std=Cityscapes.std),
        Tensor()
    ])

    dataset = Cityscapes(Path(root_path), transforms=transforms, subset=subset)

    if num_images and num_images < len(dataset):
        indices = list(range(num_images))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate)

    return loader


def save_segmentation_result(predictions, output_path):
    """
    Save segmentation predictions as a colored image using SwiftNet's colorization
    
    Args:
        predictions: Numpy array of class predictions (H, W)
        output_path: Path to save the output image
    """
    color_info = Cityscapes.color_info
    to_color = ColorizeLabels(color_info)

    colored_pred = to_color(predictions)
    colored_pred_img = Image.fromarray(colored_pred)
    colored_pred_img.save(output_path)


def save_comparison_visualization(original_image, prediction, output_path):
    """
    Save a side-by-side visualization of original image and segmentation
    
    Args:
        original_image: PIL Image or numpy array
        prediction: Numpy array of class predictions (H, W)
        output_path: Path to save the output image
    """
    # Convert original image to numpy if it's a PIL image
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)
    
    # Ensure original image is RGB and correct format
    if original_image.ndim == 3 and original_image.shape[0] == 3:
        # Convert CHW to HWC
        original_image = np.transpose(original_image, (1, 2, 0))
    
    # Convert segmentation to RGB using SwiftNet's colorization
    color_info = Cityscapes.color_info
    to_color = ColorizeLabels(color_info)
    seg_rgb = to_color(prediction)
    
    # Get dimensions
    h, w = prediction.shape
    
    # Resize original image if needed
    if original_image.shape[:2] != (h, w):
        original_image = np.array(Image.fromarray(original_image).resize((w, h)))
    
    # Create side-by-side visualization
    combined = np.hstack([original_image, seg_rgb])
    Image.fromarray(combined).save(output_path)
