#!/usr/bin/env python


import rospy
from sensor_msgs.msg import Image
from utils import cv2_to_ros_image
import cv2


class VideoPublisher:
    def __init__(self, video_file_path,interval=2):
        self.video_file_path = video_file_path
        self.publisher = rospy.Publisher('/hik_cam_node/hik_camera', Image, queue_size=1)
        self.video_capture = cv2.VideoCapture(video_file_path)
        self.interval=interval
        self.rate = self.video_capture.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

        
    def publish_frames(self):
        rate = rospy.Rate(self.rate)  # 0.5秒间隔
        
        idx_frame=0
        while not rospy.is_shutdown() and self.video_capture.isOpened():
            while self.video_capture.grab():
                #t = time.time()
                _, frame = self.video_capture.retrieve()
                if idx_frame % self.interval == 0:
                    print("publishing frame "+str(idx_frame), end='\r')
                    ros_image = cv2_to_ros_image(frame, rospy.Time.now())
                    self.publisher.publish(ros_image)
                rate.sleep()
                idx_frame+=1
        self.video_capture.release()


if __name__ == '__main__':
    try:
        rospy.init_node('video_publisher', anonymous=True)  
        video_file_path=rospy.get_param('video_path','/home/user/Mask2Anomaly/data/video1.mp4')
        interval=rospy.get_param('interval',1)
        video_publisher = VideoPublisher(video_file_path,interval)
        video_publisher.publish_frames()
    except rospy.ROSInterruptException:
        pass
