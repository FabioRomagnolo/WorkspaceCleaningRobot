import torch
import os
import cv2
from torchvision import transforms as T
from ..model import MattingBase, MattingRefine

AVAILABLE_BACKBONES = ['mobilenetv2', 'resnet50']
SAMPLE_IMAGES_DIR = 'sample_images'


def load_inference_model(path_to_model,
                         model_type='mattingrefine',
                         input_resolution='hd',
                         refine_mode='sampling',
                         device=torch.device("cpu"),
                         verbose=True):
    """
    Method to load specific model
    :param model_type: Model type available are:
                        - "mattingbase", trained on low resolution images.
                        - "mattingrefine", trained on high resolution details.
    :param input_resolution: Expected input resolution. Can be:
                        - 'nhd'
                        - 'hd'
                        - '4k'.
    :param refine_mode: Refinement area selection mode. Options:
                        - "full", refine everywhere using regular Conv2d, but it's not recommended (too many noises).
                        - "sampling", refine fixed amount of pixels ranked by the top most errors.
                        - "thresholding", refine varying amount of pixels that have greater error than the threshold.
    :param device: Torch device on which loading model.
    :param verbose:
    :return: Model instance
    """
    assert model_type in ['mattingbase', 'mattingrefine']
    assert refine_mode in ['full', 'sampling', 'thresholding']
    assert input_resolution in ['nhd', 'hd', '4k']
    model = None

    # Identifying model backbone by name of file
    model_backbone = None
    for backbone in AVAILABLE_BACKBONES:
        if backbone in os.path.basename(path_to_model):
            model_backbone = backbone
            break
    if not model_backbone:
        raise f"ERROR! The PyTorch file should relate to one of these backbones: {AVAILABLE_BACKBONES}"

    # Loading PyTorch model
    if model_type == 'mattingbase':
        model = MattingBase(backbone=model_backbone, inference=True).to(device)
    elif model_type == 'mattingrefine':
        model = MattingRefine(backbone=model_backbone, inference=True,
                              input_resolution=input_resolution,
                              refine_mode=refine_mode).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device), strict=False)
    model = model.eval()
    return model


def image_file_to_tensor(path_to_file, precision, device):
    # Read the image
    image = cv2.imread(path_to_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a transform to convert the image to tensor
    tensor = T.ToTensor()(image)
    # Convert the image to PyTorch tensor according to requirements
    return tensor.to(precision).to(device)


def get_dummy_inputs(resolution='hd', precision=torch.float32, device='cpu'):
    """
    Get dummy inputs for the Background Matting network.
    :param resolution: Resolution of the dummy inputs. It can be:
                       - nhd
                       - hd
                       - 4k
    :param precision: Precison of returned tensor. It can be:
                      - torch.float16
                      - torch.float32
    :param device: Device on which load the returned tensors. It can be:
                      - 'cpu'
                      - 'cuda'.
    :return Dummy src, bgr
    """
    path_to_src, path_to_bgr = None, None
    if resolution == 'nhd':
        path_to_src = os.path.join(SAMPLE_IMAGES_DIR, 'sample_src_nhd.jpg')
        path_to_bgr = os.path.join(SAMPLE_IMAGES_DIR, 'sample_bgr_nhd.jpg')
    if resolution == 'hd':
        path_to_src = os.path.join(SAMPLE_IMAGES_DIR, 'sample_src_hd.jpg')
        path_to_bgr = os.path.join(SAMPLE_IMAGES_DIR, 'sample_bgr_hd.jpg')
    if resolution == '4k':
        path_to_src = os.path.join(SAMPLE_IMAGES_DIR, 'sample_src_4k.jpg')
        path_to_bgr = os.path.join(SAMPLE_IMAGES_DIR, 'sample_bgr_4k.jpg')

    if path_to_src and path_to_bgr:
        src = image_file_to_tensor(path_to_src, precision, device)
        bgr = image_file_to_tensor(path_to_bgr, precision, device)
        return src.unsqueeze(0), bgr.unsqueeze(0)
    else:
        raise "ERROR. Resolution parameter can be only: nhd, hd, 4k"
