# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import visualize, reconstruct

def fig_2_cv2(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def predict_3D_pose(bgr_img, pose, depth, width, height, multiview=False):
    resize_img = cv2.resize(bgr_img, (width, height))
    img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
     #predict keypoints
    counts, objects, peaks = pose.predict(img)
    
    #predict depth 
    depth_map = depth.predict(img)
    
    #draw keypoints
    all_keypoints = reconstruct.get_keypoints(objects, counts, peaks, height=height, width=width)
    image = visualize.draw_2d(resize_img, all_keypoints)
    all_xyz = reconstruct.to_xyz(all_keypoints, depth_map)
    
    #show in 3d
    fig = visualize.draw_3d(image, all_xyz, multiview=multiview)
    image = fig_2_cv2(fig)
    plt.close(fig)
    
    return image
