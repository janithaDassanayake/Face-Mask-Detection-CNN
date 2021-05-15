# python face_count.py

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

face_count = 0
face_detect = False
time_stamp = 0
look = False
intervel = 1  # change time
mask_status = False
sanitizer_count = 0

time_stamp = time.time()


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    if (len(locs) > 0):
        face_detect = True

    else:
        face_detect = False

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        if (mask > withoutMask):
            mask_status = True
        else:
            mask_status = False

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    if (face_detect == True):
        time_stamp = time.time()

    if (time_stamp + intervel > time.time()):

        if (look == False):
            look = True
            face_count = face_count + 1
            print(face_count, ' -  mask_status = ', mask_status)
    else:
        look = False
        sanitizer_count = 0
    # pass

    cv2.putText(frame, 'Face count = ' + str(face_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    cv2.putText(frame, 'sanitizer count = ' + str(sanitizer_count), (230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 0, 0), 2)

    if sanitizer_count > 0:
        if (mask_status == True):
            cv2.putText(frame, 'Access Granted !', (65, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        else:
            cv2.putText(frame, 'Pleas Wear The Mask', (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 0, 0), 2)
    else:
        cv2.putText(frame, 'Please Senitize Your Hands !', (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 0, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("z"):
        sanitizer_count = sanitizer_count + 1
        time.sleep(0.3)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
