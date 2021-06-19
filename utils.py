import matplotlib.pyplot as plt
import math
import torchvision
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import cv2 as cv
import torch
from PIL import Image
from SimCLR import SimCLR
import torch.nn as nn
import numpy as np

def cam(file_path):
    simclr = SimCLR([1])
    model = simclr.model
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load('classifier.pt'))
    model.to('cpu')


    gradcam = True
    if gradcam:

        from torchcam.cams import SmoothGradCAMpp, CAM, GradCAM, ScoreCAM, GradCAMpp, SSCAM, ISCAM
        image = cv.imread(file_path)
        zero = np.zeros_like(image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        zero[:, :, 0] = gray
        zero[:, :, 1] = gray
        zero[:, :, 2] = gray
        image = zero

        img = image
        img = img[:224, :224, :]
        pil_img = Image.fromarray(img)

        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(dim=0)
        print(img.shape)
        img_tensor = torchvision.transforms.Normalize((127.0710, 127.0710, 127.0710), (71.4902, 71.4902, 71.4902))(img)

        # Hook the corresponding layer in the model
        cam_extractors = [
            CAM(model, fc_layer='fc'),

        ]

        # Don't trigger all hooks
        for extractor in cam_extractors:
            extractor._hooks_enabled = False

        num_rows = 2
        num_cols = math.ceil(len(cam_extractors) / num_rows)

        class_idx = None
        for idx, extractor in enumerate(cam_extractors):
            extractor._hooks_enabled = True
            model.zero_grad()
            scores = model(img_tensor)

            # Select the class index
            class_idx = scores.squeeze(0).argmax().item() if class_idx is None else class_idx
            print(class_idx)
            # Use the hooked data to compute activation map
            activation_map = extractor(class_idx, scores).cpu()

            # Clean data
            extractor.clear_hooks()
            extractor._hooks_enabled = False
            # Convert it to PIL image
            # The indexing below means first image in batch
            heatmap = to_pil_image(activation_map, mode='F')
            # Plot the result
            result = overlay_mask(pil_img, heatmap)
            plt.imshow(result)

            plt.title(extractor.__class__.__name__)

        plt.tight_layout()
        if True:
            plt.savefig('result' + str(idx), dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()
