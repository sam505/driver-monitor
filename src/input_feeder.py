import cv2
import mss
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class InputFeeder:
    def __init__(self, input_type, input_file=None):
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image':
            self.input_file = input_file
    
    def load_data(self):
        if self.input_type == 'video':
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.imread(self.input_file)

    def next_batch(self):
        while self.cap.isOpened():
            start = time.time()
            frames = 0
            for i in range(10):
                frames += 1
                fps = frames / (time.time() - start)
                logger.info('Actual video fps: {}'.format(int(fps)))
                _, frame = self.cap.read()

                if _ is not True:
                    break

            return frame

    def close(self):
        if not self.input_type == 'image':
            self.cap.release()

