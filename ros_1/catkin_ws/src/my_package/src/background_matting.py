import torch
import os
import time
from BackgroundMatting.utils.model_utils import load_inference_model, get_dummy_inputs
from torchvision.transforms.functional import to_pil_image

MODELS_DIR = os.path.join('BackgroundMatting', 'trained_models')
MODEL_PATHS = {
    'mobilenetv2': os.path.join(MODELS_DIR, 'pytorch_mobilenetv2.pth'),
    'resnet50': os.path.join(MODELS_DIR, 'pytorch_resnet50.pth'),
}


class BackgroundMatting:
    def __init__(self,
                 backbone='mobilenetv2',
                 model_type='mattingrefine',
                 input_resolution='nhd',
                 device=torch.device('cpu')):
        self.model = load_inference_model(
            path_to_model=MODEL_PATHS[backbone],
            model_type=model_type,
            input_resolution=input_resolution,
            device=device,
        )

    def matting(self, src, bgr):
        """
        :param src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        :param bgr: (B, 3, H, W) the background image. Channels are RGB values normalized to 0 ~ 1.
        :return: (B, 4, H, W) the transparent foreground prediction (foreground + alpha).
                 Channels are RGBA values normalized to 0 ~ 1.
        """
        with torch.no_grad():
            tfgr = self.model(src, bgr)
        return tfgr


def dummy_test(model_type='mattingrefine', resolution='nhd'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"- Loading model {model_type} on {device} ...")

    matting = BackgroundMatting(model_type=model_type,
                                input_resolution=resolution,
                                device=device)
    src, bgr = get_dummy_inputs(resolution=resolution, device=device)

    # Inference
    start_time = time.time()
    tfgr = matting.matting(src, bgr)
    inference_time = time.time() - start_time
    print(f"Model took up {inference_time} seconds on {device} to inference!")

    # Showing result
    pil_tfgr = to_pil_image(tfgr[0].cpu())
    pil_tfgr.show()
    return tfgr

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("WARNING! Unable to use the GPU for inference. This will slow down performances ...")
    # Testing with dummy images
    dummy_test()
