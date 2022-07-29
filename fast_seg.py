import torch
from PIL import Image
from IPython.display import display

from fastseg import MobileV3Large
from fastseg.image import colorize, blend


def get_fastseg_model():
    model = MobileV3Large.from_pretrained()
    return model

def get_colorized_seg_map(image):
    return colorize(image)

# model = get_fastseg_model()
# x = torch.random(())