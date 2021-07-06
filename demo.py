from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

model_store_dir = "models/classifier.model"
maskNet = load_model(model_store_dir)

for folders in os.listdir('./dataset2'):
    folderpath = os.path.join('./dataset2', folders)
    for imgName in os.listdir(folderpath):
        imgpath = os.path.join(folderpath, imgName)
        # print(os.listdir(folderpath))
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (720, 720))
        copy = cv2.resize(image.copy(), (224, 224))

        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        copy = cv2.resize(copy, (224, 224))
        copy = img_to_array(copy)
        copy = preprocess_input(copy)
        # Expand dimentions - converting 1D to 2D
        copy = np.expand_dims(copy, axis=0)

        (mask, without_mask) = maskNet.predict(copy)[0]

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)
        cv2.putText(image, label, (500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow('image', image)

        # pauses for 2 seconds before fetching next image
        key = cv2.waitKey(2000)
        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
