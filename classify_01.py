# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg
# python3 classify.py --model ab.model --labelbin ablb.pickle --image test_image_random/2_test.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time

def getlabel(idxs):
	# https://www.geeksforgeeks.org/enumerate-in-python/
	for (i, j) in enumerate(idxs):
		# build the label and draw the label on the image
		#label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
		label = mlb.classes_[j]
		if i==0:
			return label

def getresult(file_image, l_test_ir):
	for test_ir in l_test_ir:
		test_ir = test_ir.split(';')
		if test_ir[0] == file_image:
			return test_ir[1]
#start time
start = time.time()

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help="path directory image")
ap.add_argument("-t", "--test", required=True, help="file test image random")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())
folder = args["image"]
#l_test_ir = list(open(args["test"],"r"))
# https://stackoverflow.com/questions/3925614/how-do-you-read-a-file-into-a-list-in-python
with open(args["test"]) as f:
	l_test_ir = f.read().splitlines()

print("[INFO] classifying image...")
path, dirs, files = next(os.walk(folder))
nb_files = len(files)
same = 0
for i in range(nb_files):
	file_image = files[i]
	image = cv2.imread(folder+'/'+file_image)
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	proba = model.predict(image)[0]
	idxs = np.argsort(proba)[::-1][:2]
	label = getlabel(idxs)
	result = getresult(file_image,l_test_ir)
	if label == result :
		same = same + 1
	else :
		print("Tidak match : File {0} P : {1} \t\t R : {2}".format(file_image,label,result))
accuracy = same / nb_files
print("Hasil accuracy : {:.2f}".format(accuracy*100))
end = time.time()
# https://pythonhow.com/measure-execution-time-python-code/
print("time running : {0}".format(end-start))
	#print(file_image+'/'+label)
# # load the image
# image = cv2.imread(args["image"])
# output = imutils.resize(image, width=400)
#
# # pre-process the image for classification
# image = cv2.resize(image, (96, 96))
# image = image.astype("float") / 255.0
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
#
# # load the trained convolutional neural network and the multi-label
# # binarizer
# print("[INFO] loading network...")
# model = load_model(args["model"])
# mlb = pickle.loads(open(args["labelbin"], "rb").read())
#
# # classify the input image then find the indexes of the two class
# # labels with the *largest* probability
# print("[INFO] classifying image...")
# proba = model.predict(image)[0]
# idxs = np.argsort(proba)[::-1][:2]
#
# # loop over the indexes of the high confidence class labels
# for (i, j) in enumerate(idxs):
# 	# build the label and draw the label on the image
# 	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
# 	cv2.putText(output, label, (10, (i * 30) + 25),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
# # show the probabilities for each of the individual labels
# for (label, p) in zip(mlb.classes_, proba):
# 	print("{}: {:.2f}%".format(label, p * 100))
#
# # show the output image
# cv2.imshow("Output", output)
# cv2.waitKey(0)