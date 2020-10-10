from openvino.inference_engine import IENetwork, IECore
import cv2
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class HeadPoseEstimation:
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
        input_name = next(iter(self.net.inputs))
        output_names = self.net.outputs
        input_shape = self.net.inputs[input_name].shape
        output_shapes = []
        for name in output_names:
            output_shape = self.net.outputs[name].shape
            output_shapes.append(output_shape)

        return input_name, input_shape, output_names, output_shapes
        raise NotImplementedError

    def preprocess_input(self, image):
        input_name, input_shape, output_names, output_shapes = self.check_model()
        image = cv2.resize(image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image
        raise NotImplementedError

    def predict(self, image):
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_names, output_shapes = self.check_model()
        input_dict = {input_name: processed_image}
        self.net.start_async(request_id=0, inputs=input_dict)
        results = []
        if self.net.requests[0].wait(-1) == 0:
            for name in output_names:
                result = self.net.requests[0].outputs[name]
                results.append(result)

        return results
        raise NotImplementedError

    def preprocess_output(self, image):
        results = self.predict(image)
        values = []
        for i in range(len(results)):
            values.append(results[i][0][0])

        return values
        raise NotImplementedError