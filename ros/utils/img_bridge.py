import numpy as np
from sensor_msgs.msg import Image
import cv2

def ros_image_to_cv2(ros_image):
    # Decode from sensor_msgs/Image to CV2 image
    dtype = np.dtype("uint8")  # Assuming 8-bit image data
    image = np.frombuffer(ros_image.data, dtype=dtype)
    image = image.reshape(ros_image.height, ros_image.width, -1)  # Assuming image is not compressed
    # Convert from RGB to BGR if necessary
    if 'rgb' in ros_image.encoding.lower():
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def cv2_to_ros_image(cv_image, timestamp, encoding="bgr8"):
    # Encode from CV2 image to sensor_msgs/Image
    ros_image = Image()
    ros_image.header.stamp = timestamp
    ros_image.height = cv_image.shape[0]
    ros_image.width = cv_image.shape[1]
    ros_image.encoding = encoding
    ros_image.is_bigendian = 0
    ros_image.step = cv_image.shape[1] * cv_image.shape[2]
    ros_image.data = cv_image.tostring()
    return ros_image
