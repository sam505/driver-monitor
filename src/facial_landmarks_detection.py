from openvino.inference_engine import IENetwork, IECore
import cv2
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class FacialLandmarks:
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name + '.xml'
        self.model_structure = model_name + '.bin'
        self.device = device
        self.net = None
        self.count = 0

    def load_model(self):
        core = IECore()
        start = time.time()
        model = core.read_network(self.model_weights, self.model_structure)
        logger.info('Loading the Facial Landmarks Detection Model...')
        self.net = core.load_network(network=model, device_name=self.device, num_requests=1)
        logger.info('Time taken to load the model is: {:.4f} seconds'.format(time.time() - start))

        return self.net
        raise NotImplementedError

    def check_model(self):
        input_name = next(iter(self.net.inputs))
        input_shape = self.net.inputs[input_name].shape
        output_name = next(iter(self.net.outputs))
        output_shape = self.net.outputs[output_name].shape

        return input_name, input_shape, output_name, output_shape
        raise NotImplementedError

    def preprocess_input(self, image):
        input_name, input_shape, output_name, output_shape = self.check_model()
        # if image.any():
        image = cv2.resize(image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image
        raise NotImplementedError

    def predict(self, image):
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name, output_shape = self.check_model()
        input_dict = {input_name: processed_image}
        self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            results = self.net.requests[0].outputs[output_name]

        return results
        raise NotImplementedError

    def preprocess_output(self, face):
        outputs = self.predict(face)
        h, w, c = face.shape
        x0, y0 = int(w * outputs[0][0][0][0]), int(h * outputs[0][1][0][0])
        x1, y1 = int(w * outputs[0][2][0][0]), int(h * outputs[0][3][0][0])
        factor = x1 - x0
        face = cv2.rectangle(face, (x0 - int(0.3*factor), y0 - int(0.2*factor)),
                             (x0 + int(0.3*factor), y0 + int(0.2*factor)), (255, 255, 0), 1)
        face = cv2.rectangle(face, (x1 - int(0.3*factor), y1 - int(0.2*factor)),
                             (x1 + int(0.3*factor), y1 + int(0.2*factor)), (255, 255, 0), 1)
        cv2.imshow('Facial Landmarks', face)
        cv2.waitKey(1)
        left_eye = face[y0 - int(0.2*factor): y0 + int(0.2*factor), x0 - int(0.3*factor):x0 + int(0.3*factor)]
        right_eye = face[y1 - int(0.2*factor): y1 + int(0.2*factor), x1 - int(0.3*factor):x1 + int(0.3*factor)]

        return right_eye, left_eye
        raise NotImplementedError
