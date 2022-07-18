import torch
import os
import time
import argparse
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image, to_tensor
from BackgroundMatting.utils.model_utils import load_inference_model, get_dummy_inputs
from torchvision.transforms.functional import to_pil_image

MODELS_DIR = os.path.join('BackgroundMatting', 'trained_models')
MODEL_PATHS = {
    'mobilenetv2': os.path.join(MODELS_DIR, 'pytorch_mobilenetv2.pth'),
    'resnet50': os.path.join(MODELS_DIR, 'pytorch_resnet50.pth'),
}

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Background Matting test')
parser.add_argument('--model-type', type=str, default='mattingrefine', choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, default='resnet50', choices=['resnet50', 'mobilenetv2'])
parser.add_argument('--input-resolution', type=str, default='nhd', choices=['nhd', 'hd', '4k'])

args = parser.parse_args()


class BackgroundMatting:
    def __init__(self,
                 backbone='resnet50',
                 model_type='mattingrefine',
                 refine_mode='sampling',
                 input_resolution='nhd',
                 device=torch.device('cpu')):
        self.model = load_inference_model(
            path_to_model=MODEL_PATHS[backbone],
            model_type=model_type,
            refine_mode=refine_mode,
            input_resolution=input_resolution,
            device=device,
        )

    def preproccesing(self, *tensors):
        h, w = 360, 640     # NHD resolution
        if self.model.input_resolution == 'hd':
            h, w = 720, 1280
        elif self.model.input_resolution == '4k':
            h, w = 2160, 3840

        for t in tensors:
            # Converting to PIL image
            pic = to_pil_image(t.squeeze(0))
            # Resizing pic
            resized_pic = T.Resize((h, w))(pic)
            # Converting back to Torch tensor
            t = to_tensor(resized_pic).unsqueeze(0)
        return tensors

    def matting(self, src, bgr):
        """
        :param src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        :param bgr: (B, 3, H, W) the background image. Channels are RGB values normalized to 0 ~ 1.
        :return: (B, 4, H, W) the transparent foreground prediction (foreground + alpha).
                 Channels are RGBA values normalized to 0 ~ 1.
        """
        # Preprocessing
        src, bgr = self.preproccesing(src, bgr)
        # Inference
        with torch.no_grad():
            tfgr = self.model(src, bgr)
        return tfgr


def dummy_test(model_type='mattingrefine', backbone='resnet50', resolution='nhd'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"- Loading model {model_type} with backbone {backbone} on {device} ...")

    model = BackgroundMatting(model_type=model_type,
                                backbone=backbone,
                                input_resolution=resolution,
                                device=device)
    src, bgr = get_dummy_inputs(resolution=resolution, device=device)

    # Preprocessing + inference
    start_time = time.time()
    tfgr = model.matting(src, bgr)
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
    dummy_test(model_type=args.model_type, backbone=args.model_backbone, resolution=args.input_resolution)
