import json
import torch
import torch2trt
import numpy as np
import torchvision.transforms as transforms
from trt_pose import models, coco
from PIL import Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

class Pose:
    """
    2D pose estimation class
    """
    def __init__(self, args, model_type='trt', backbone='densenet121'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.width, self.height = (256, 256) if backbone=='densenet121' else (224, 224)
        
        # load humanpose json data
        self.meta = self.load_json(args.json)
        # load trt model
        if model_type == 'trt':
            self.model  = self._load_trt_model(args.model_file)         
        elif model_type == 'torch':
            self.model = self._load_torch_model(args.model_file, backbone=backbone)     
        else:
            print('not supported model type "{}"'.format(model_type))
            return -1

        self.topology = coco.coco_category_to_topology(self.meta)
        self.parse_objects = ParseObjects(self.topology, cmap_threshold=0.08, link_threshold=0.08)
        self.draw_objects = DrawObjects(self.topology)

        # transformer
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    @staticmethod
    def load_json(json_file):
        with open(json_file, 'r') as f:
            meta = json.load(f)
        return meta

    def _load_trt_model(self, model_file):
        """
        load converted tensorRT model  
        """
        print('[INFO] Loading tensorrt pose model')
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(model_file))
        model_trt.eval()
        return model_trt

    def _load_torch_model(self, model_file, backbone='densenet121'):
        """
        load pytorch model with resnet18 encoder or densenet121
        """ 
        print('[INFO] Loading pytorch 2d_pose model with "{}"'.format(backbone.title()))
        num_parts = len(self.meta['keypoints'])
        num_links = len(self.meta['skeleton'])
        
        if backbone=='resnet18':
            model = models.resnet18_baseline_att(cmap_channels=num_parts,
                                                 paf_channels=2 * num_links) 
        elif backbone=='densenet121':
            model = models.densenet121_baseline_att(cmap_channels=num_parts,
                                                    paf_channels= 2 * num_links)
        else:
            print('not supported model type "{}"'.format(backbone))
            return -1
            
        model.to(self.device).eval()
        model.load_state_dict(torch.load(model_file))
        return model 
    
    def predict(self, image: np.ndarray):
        """
        predict pose estimation on rgb array image
        *Note - image need to be RGB numpy array format
        """
        assert isinstance(image, np.ndarray), 'image must be numpy array'
        pil_img, tensor_img = self.preprocess(image)
        cmap, paf = self.model(tensor_img)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf) # cmap threhold=0.15, link_threshold=0.15
        # print('[INFO] Numbers of person detected : {} '.format(counts.shape[0]))
        return counts, objects, peaks
    
    def preprocess(self, image):
        """
        resize image and transform to tensor image
        """
        assert isinstance(image, np.ndarray), 'image type need to be array'
        image = Image.fromarray(image).resize((self.width, self.height), 
                                              resample=Image.BILINEAR)
        tensor = self.transforms(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        return image, tensor


    

       
        
    
        
                        