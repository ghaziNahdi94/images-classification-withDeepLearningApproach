import keras
import cv2
import numpy as np






#pretrained model
model = keras.models.load_model("./model.h5")


classes = ['fish','humain','laptop','backpack','dog','billiard']


#path to image for prediction
file="./5.jpeg"

image = cv2.imread(file)
image = cv2.resize(image, (139, 139), 0, 0, cv2.INTER_LINEAR)
image = np.multiply(image, 1.0 / 255.0)

images = []

images.append(image)

images = np.array(images)

predictions = model.predict(images)

pred = predictions[0]

k = []

for i in pred :
    k.append(i)

max = k.index(max(k))

print(classes[max])


