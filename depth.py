# -*- coding: utf-8 -*-
import cv2
import argparse
# from app import APP
import torch
from PIL import Image
from torchvision import transforms
from models.bts import BtsModel
import numpy as np

class Depth():
    _defaults = {
        "encoder" : 'resnet50',
        "bts_size" : 512
        }

    def __init__(self, args, **kwargs):
        self.args = argparse.Namespace(**args, **self._defaults)
        self.args.__dict__.update(kwargs) # update kwargs value
        focal = [518.8579] if self.args.dataset=='nyu' else [715.0873]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.focal = torch.autograd.Variable(torch.tensor(focal)).to(self.device)

    def _load_model(self):
        print('[INFO] Loading depth estimation model')
        model = BtsModel(self.args)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(self.args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        model.eval().to(self.device)
        return model

    def predict(self, image):
        with torch.no_grad():
            # image = cv2.resize(image, (self.args.width, self.args.height))
            tensor = self.to_tensor(image)
            *_, depth = self.model(tensor, self.focal)
            depth = depth.cpu().numpy().squeeze()

        return depth

    def to_tensor(self, image:Image):

        assert isinstance(image, np.ndarray), 'image type need to be array'
        image = Image.fromarray(image).resize((self.args.width, self.args.height),
                                              resample=Image.BILINEAR)
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return tfms(image).unsqueeze(0).to(self.device)

    def show_depth(self, depth):
        # scaling to 255
        depth = depth * 256 if self.args.dataset=='kitti' else depth * 1000
        color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.0355), cv2.COLORMAP_JET)
        cv2.imshow('Depth Prediction', color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
