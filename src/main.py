import cv2
from src.face_detection import FaceDetection
from src.facial_landmarks_detection import FacialLandmarks
from src.head_pose_estimation import HeadPoseEstimation
from src.gaze_estimation import GazeEstimation
from playsound import playsound
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    audio = "bin/beep-01a.mp3"
    fd = FaceDetection(model_name='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-'
                                  'binary-0001')
    fl = FacialLandmarks(model_name='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-'
                                    'retail-0009')
    hpe = HeadPoseEstimation(model_name='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation'
                                        '-adas-0001')
    ge = GazeEstimation(model_name='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002')

    fd.load_model()
    fl.load_model()
    hpe.load_model()
    ge.load_model()
    count = 0
    total_score = 0

    cap = cv2.VideoCapture('/home/sammie/PycharmProjects/driver\'s_attention/bin/Guardian_driver1.mp4')
    # cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        count += 1
        if not _:
            break
        frame = cv2.flip(frame, 1)
        face = fd.preprocess_output(frame)
        try:
            image = cv2.resize(frame, (1280, 700), interpolation=cv2.INTER_AREA)
            right_eye, left_eye = fl.preprocess_output(face)
            angles = hpe.preprocess_output(face)
            gaze = ge.preprocess_output(right_eye, left_eye, angles)
            if count % 30 == 0:
                count = 1
                total_score = 0
            if -20 > angles[0] or angles[0] > 20:
                result = 0
                total_score += result
            else:
                result = 1
                total_score += result

            if -20 > angles[1] or angles[1] > 20:
                result = 0
                total_score += result
            else:
                result = 1
                total_score += result

            if -20 > angles[2] or angles[2] > 20:
                result = 0
                total_score += result
            else:
                result = 1
                total_score += result

            if gaze[0][0] < -15 or gaze[0][0] > 30:
                result = 0
                total_score += result
            else:
                result = 1
                total_score += result

            if gaze[0][1] < -10 or gaze[0][1] > 10:
                result = 0
                total_score += result
            else:
                result = 1
                total_score += result

            total_score_value = total_score/(count * 5)
            if total_score_value < 0.9:
                playsound(audio)
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            image = cv2.rectangle(image, (0, 50), (250, 150), color, cv2.FILLED)
            image = cv2.putText(image, 'Score: {:.2f}%'.format(total_score_value * 100),
                                (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)
            cv2.imshow('Head Pose', image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        except cv2.error:
            logger.info("No face detected")
            if count % 30 == 0:
                count = 1
                total_score = 0
            total_score += 0
            total_score_value = total_score / (count * 5)
            image = cv2.resize(frame, (1280, 700), interpolation=cv2.INTER_AREA)
            if total_score_value < 0.9:
                playsound(audio)
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            image = cv2.rectangle(image, (0, 50), (250, 150), color, cv2.FILLED)
            image = cv2.putText(image, 'Score: {:.2f}%'.format(total_score_value * 100),
                                (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 0), 2)

            cv2.imshow('Head Pose', image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
