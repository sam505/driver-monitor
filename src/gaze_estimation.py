from openvino.inference_engine import IENetwork, IECore
import cv2
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class GazeEstimation:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name + '.xml'
        self.model_structure = model_name + '.bin'
        self.device = device
        self.net = None

        return
        raise NotImplementedError

    def load_model(self):
        core = IECore()
        start = time.time()
        logger.info("Loading the Head Pose Estimation Model...")
        model = core.read_network(self.model_weights, self.model_structure)
        self.net = core.load_network(network=model, device_name=self.device, num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time() - start))

        return self.net
        raise NotImplementedError

    def check_model(self):
        input_names = self.net.inputs
        output_names = self.net.outputs
        input_names_list = []
        output_names_list = []
        input_shapes = []
        output_shapes = []
        for name in input_names:
            input_names_list.append(name)
            input_shape = self.net.inputs[name].shape
            input_shapes.append(input_shape)
        for name in output_names:
            output_names_list.append(name)
            output_shape = self.net.outputs[name].shape
            output_shapes.append(output_shape)

        return input_names_list, input_shapes, output_names_list, output_shapes
        raise NotImplementedError

    def preprocess_input(self, coordinates, right_eye, left_eye):
        right_eye = cv2.resize(right_eye, (60, 60), interpolation=cv2.INTER_AREA)
        right_eye = right_eye.transpose((2, 0, 1))
        right_eye = right_eye.reshape(1, *right_eye.shape)

        left_eye = cv2.resize(left_eye, (60, 60), interpolation=cv2.INTER_AREA)
        left_eye = left_eye.transpose((2, 0, 1))
        left_eye = left_eye.reshape(1, *left_eye.shape)

        coordinates = np.array(coordinates).reshape(1, 3)

        return coordinates, right_eye, left_eye
        raise NotImplementedError

    def predict(self, coordinates, left_eye, right_eye):
        coordinates, right_eye, left_eye = self.preprocess_input(coordinates, right_eye, left_eye)
        input_name, input_shape, output_names, output_shapes = self.check_model()
        input_dict = {input_name[0]: coordinates, input_name[1]: left_eye, input_name[2]: right_eye}
        self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            result = self.net.requests[0].outputs[output_names[0]]

        return result
        raise NotImplementedError

    def preprocess_output(self, right_eye, left_eye, coordinates):
        results = self.predict(coordinates, right_eye, left_eye)

        return results
        raise NotImplementedError