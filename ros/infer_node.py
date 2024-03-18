#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from utils import ros_image_to_cv2, cv2_to_ros_image
import logging

class SemanticSegmentationNode:
    def __init__(self):
        # Setup logger for Detectron2
        setup_logger()
        setup_logger(name="mask2anomaly")

        # Configuration setup for the model
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_maskformer2_config(self.cfg)
        self.cfg.merge_from_file("configs/cityscapes/semantic-segmentation/anomaly_inference.yaml")
        self.cfg.MODEL.WEIGHTS = 'checkpoints/model_final6.pth'
        self.predictor = DefaultPredictor(self.cfg)

        # ROS Node, Publisher setup
        rospy.init_node('semantic_segmentation_node')
        self.pub = rospy.Publisher('/segmentation_output', Image, queue_size=10)
        rospy.Subscriber("/hik_cam_node/hik_camera", Image, self.callback)
        self.logger = logging.getLogger("mask2anomaly")
        self.logger.info(f"Semantic Segmentation Node Initialized")



    def pred(self, im):

        # Convert image to RGB
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        outputs = self.predictor(im)

        # Original logic for anomaly segmentation predictions
        anomaly_map = outputs["sem_seg"].unsqueeze(0)
        anomaly_map = 1 - torch.max(anomaly_map[0:19,:,:], axis = 1)[0]
        if outputs["sem_seg"][19:,:,:].shape[0] > 1:
            outputs_na_mask = torch.max(outputs["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
            outputs_na_mask[outputs_na_mask < 0.5] = 0
            outputs_na_mask[outputs_na_mask >= 0.5] = 1
            outputs_na_mask = 1 - outputs_na_mask
            anomaly_map = anomaly_map*outputs_na_mask.detach()
        anomaly_map = anomaly_map.detach().cpu().numpy().squeeze().squeeze()

        anomaly_map[anomaly_map < 0.95] = 0
        anomaly_map_normalized = (anomaly_map * 255.0).astype(np.uint8)
        colored_map = cv2.applyColorMap(anomaly_map_normalized, cv2.COLORMAP_JET)
        colored_map_rgb = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)

        # Blending the original image with the anomaly map
        custom_alpha = 0.5
        mask = anomaly_map_normalized > 0
        alpha_channel = np.zeros_like(anomaly_map_normalized, dtype=np.float32)
        alpha_channel[mask] = custom_alpha * 255
        im_rgba = np.dstack((im_rgb, np.ones(im_rgb.shape[:2], dtype=np.uint8) * 255))
        alpha_channel_3d = np.repeat(alpha_channel[:, :, np.newaxis], 4, axis=2)
        alpha_channel_3d[..., 3] = 255
        colored_map_rgba = np.dstack((colored_map_rgb, np.ones(colored_map_rgb.shape[:2], dtype=np.uint8) * 255))
        blended_image = im_rgba * (1 - alpha_channel_3d / 255) + colored_map_rgba * (alpha_channel_3d / 255)
        blended_image_rgb = blended_image[..., :3].astype(np.uint8)

        return blended_image_rgb

    def callback(self, data):
        cv_image = ros_image_to_cv2(data)
        segmented_image = self.pred(cv_image)
        ros_image = cv2_to_ros_image(segmented_image, rospy.Time.now())
        self.pub.publish(ros_image)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = SemanticSegmentationNode()
    node.run()
