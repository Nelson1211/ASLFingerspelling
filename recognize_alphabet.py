import glob
import sys

import cv2
import numpy as np
import os
import tensorflow as tf
from utils import detector_utils as detector_utils
from handshape_feature_extractor import HandShapeFeatureExtractor
from collections import Counter
from sklearn.metrics import classification_report

def get_inference_vector_one_frame_alphabet():
    # model trained based on https://www.kaggle.com/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

    model = HandShapeFeatureExtractor.get_instance()
    video_names = []
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    count = 0
    myFiles = glob.glob('data/*.mp4')
    alphabets = []
    sequence = []
    for file in myFiles:
        vectors = []
        sequence.append(file[-5:-4])
        # if os.path.exists('{}'.format(file[:-4])):
        #     os.remove('{}'.format(file[:-4]))
        os.mkdir('{}'.format(file[:-4]))
        cam = cv2.VideoCapture("{}".format(file))
        currentframe = 0
        path = '{}/'.format(file[:-4])
        bug = 0
        while(True):

            # reading from frame
            ret,frame = cam.read()

            if ret:
                # if video is still left continue creating images
                name = 'frame' + str(currentframe) + '.png'
                # print ('Creating...' + name)

                if (currentframe + bug) % 30 == 0:
                    # writing the extracted images
                    #cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    bug = 0
                    image_np = frame
                    image_np = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    h, w, c = image_np.shape
                    im_width = w
                    im_height = h

                    try:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
                        # while scores contains the confidence for each of these boxes.
                        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

                        boxes, scores = detector_utils.detect_objects(image_np,
                                                                    detection_graph, sess)

                        # draw bounding boxes on frame
                        cropped_image = detector_utils.draw_box_on_image(1, 0.2,
                                                        scores, boxes, im_width, im_height,
                                                        image_np)

                        img = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
                        results = model.extract_feature(img)
                        results = np.squeeze(results)
                        vectors.append(results)
                        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(path + name, cropped_image)
                    except:
                        bug += 1

                    # increasing counter so that it will
                    # show how many frames are created
                currentframe += 1
            else:
                break

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()
        alphabets.append(vectors)

    sess.close()
    return alphabets, sequence


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id

label_file = 'output_labels_alphabet.txt'
id_to_labels, labels_to_id = load_label_dicts(label_file)

prediction_vector, sequence = get_inference_vector_one_frame_alphabet()
predicted = []
for alphas in prediction_vector:
    prediction = []
    index = -1
    for values in alphas:
        counter = 0
        max_value = -1
        for value in values:
            if max_value < value:
                max_value = value
                index = counter
            counter += 1
        prediction.append(index)
    most_common,num_most_common = Counter(prediction).most_common(1)[0]
    predicted.append(id_to_labels[most_common])

with open('output_labels_alphabet.txt') as f:
    target_names = [line.strip() for line in f]
target_names = target_names[:-1]
print(classification_report(sequence, predicted, target_names=target_names))