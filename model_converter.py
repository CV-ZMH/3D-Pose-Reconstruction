# -*- coding: utf-8 -*-
import torch2trt
import torch
import argparse

def options():        
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='model_weights/human_pose.json', 
                    help='json file for pose estimation')
    ap.add_argument('--model_file', default='model_weights/densenet121_trtpose.pth', 
                    help='model file path')
    ap.add_argument('--size', required=True, type=tuple, help='pretrained model input size')
    ap.add_argument('--output', default='model_weights/resnet.pth', help='output model file')
    return ap.parse_args()
    

class ModelConverter():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_trt(self, model, out_file: str, w=224, h=224):
        """
        convert pytorch model to tensorrt
        """
        data = torch.zeros((1, 3, h, w)).to(self.device)
        print('[INFO] Converting to tensorrt')
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
        torch.save(model_trt.state_dict(), out_file)
        print('[INFO] Saved to {}'.format(out_file))

def main():
    from pose import Pose
    converter = ModelConverter()
    args = options()
    backbone = 'densenet121' if 'densenet121' in args.model_file else 'resnet18'
    pose = Pose(args, model_type='torch', backbone=backbone)    
    converter = ModelConverter()
    converter.convert_trt(pose.model, args.output, w=pose.width, h=pose.height)

if __name__ == '__main__':
 
    main()
    
        
    
    