import os
import pickle

import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


def classify(model_file="pokedex.model", label_bin="lb.pickle", image_file="examples/charmander_counter.png"):
    # load the image
    image = cv2.imread(image_file)
    output = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network and the label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(model_file)
    lb = pickle.loads(open(label_bin, "rb").read())

    # classify the input image
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]

    # we'll mark our prediction as "correct" of the input image filename
    # contains the predicted label text (obviously this makes the
    # assumption that you have named your testing image files this way)
    filename = image_file[image_file.rfind(os.path.sep) + 1:]
    correct = "correct" if filename.rfind(label) != -1 else "incorrect"

    # build the label and draw the label on the image
    label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
    output = imutils.resize(output, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # show the output image
    print("[INFO] {}".format(label))
    cv2.imshow("Output", output)
    cv2.waitKey(0)
