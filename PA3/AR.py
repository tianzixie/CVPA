from skimage import exposure
from skimage import feature
from PIL import Image
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import argparse
import numpy as np
import os
import glob
import time

def init():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dir", type=str, default="ucf_sports_actions", 
		help="path of dataset")
	ap.add_argument("-c", "--cell", type=str, default=(8, 8),
		help="pixels per cell")
	ap.add_argument("-b", "--block", type=str, default=(2, 2),
		help="cells per block")
	ap.add_argument("-bi", "--bin", type=str, default=9,
		help="HoG bin number")
	ap.add_argument("-s", "--size", type=str, default=(64, 128),
		help="resize image into (width, height)")
	ap.add_argument("-k", "--kFold", type=str, default=5,
		help="k fold cross validation")
	return vars(ap.parse_args())

def getFilePath(path):
	imgPathList = []
	txtPathList = []
	
	for filename in os.listdir(path):
		if filename == "ucf action":
			path = path + "/ucf action"
			for actionFileName in os.listdir(path):

				if (actionFileName != "Golf-Swing-Back" and 
					actionFileName != ".DS_Store" and 
					actionFileName != "Golf-Swing-Front"):
					subPath = path + "/" + actionFileName
					print("loading " + subPath)
					for splitFileNum in os.listdir(subPath):
						if splitFileNum == ".DS_Store":
							continue
						imagePath = subPath + "/" + splitFileNum
						try:
							os.listdir(imagePath + "/gt")
						except IOError:
							print(splitFileNum + " file doesn't have gt txt to crop image and recognize action.")
						else:
							for splitImageFile in os.listdir(imagePath):
								if os.path.splitext(splitImageFile)[-1] == ".jpg":
									imgPathList.append(imagePath + "/" + splitImageFile)
								if splitImageFile == "gt":
									gtPath = imagePath + "/gt"
									for splitImageGTFile in glob.glob(os.path.join(gtPath, '*.txt')):
										txtPathList.append(splitImageGTFile)
						# print("-"*100)
						# print(imagePath)
						# print(len(imgPathList))
						# print(len(txtPathList))
						# print("-"*100)
						
	return imgPathList, txtPathList

def cropImg(x, y, u, v, img):
	temp = np.zeros((v, u))
	imgArr = np.array(img)
	for y_ in range(temp.shape[0]):
		for x_ in range(temp.shape[1]):
			temp[y_][x_] = imgArr[y+y_][x+x_]
	return Image.fromarray(temp)

def HoG(imgPathList, txtPathList, size, binNum, cell, block):
	print("doing hog for all image from dataset .........")
	Y = []
	X = []
	start_time = time.time()
	count = 0
	while len(imgPathList) != 0 or len(txtPathList) != 0:
		path = imgPathList.pop()
		img = Image.open(path).convert('L')
		file = open(txtPathList.pop(), "r")
		temp = file.read().split("\t")
		x = (int)(temp[0])
		y = (int)(temp[1])
		u = (int)(temp[2])
		v = (int)(temp[3])
		if(u == 0 or v == 0):
			continue

		#	saving action into Y list
		Y.append(temp[4])

		#	crop part of image from original image
		img = cropImg(x, y, u, v, img)

		#	resize all the image into same size. As a result, all the image will have same
		#	dimensional vector and able to fit into SVM 
		img = img.resize(size)

		#	doing hog 
		HoG = feature.hog(img, orientations=binNum, pixels_per_cell=cell, 
			cells_per_block=block, block_norm="L2", transform_sqrt=True, visualise=False)

		#	saving dimensional vector into X list
		X.append(HoG.tolist())

		#	close gt file and image
		file.close()
		img.close()
	end_time = time.time()
	print("doing hog took :", str(end_time - start_time), " sec")
	return X, Y

def svcGridSearch(XTrains, XTests, YTrains, YTests, k):
	print("training model by using SVC")
	start_time = time.time()

	# k cross validation, one vs one, testing several cost penalty 
	svc = GridSearchCV(svm.SVC(kernel="linear", decision_function_shape="ovo"),
								param_grid=[ { "C":[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5] } ],
								cv=k, n_jobs=-1).fit(XTrains, YTrains)
	end_time = time.time()
	print("training model took :", str(end_time - start_time), " sec")
	return svc

def SVC(X, Y, k):
	X = np.array(X)
	Y = np.array(Y).flatten()

	#	seperate X and Y into training set and testing set
	#	the ratio of training set to testing set is 7:3
	XTrains, XTests, YTrains, YTests = train_test_split(X, Y, test_size=0.3, random_state=1)

	#	using grid search tool to test several parameters
	svc = svcGridSearch(XTrains, XTests, YTrains, YTests, k)
	return svc.best_score_, svc.best_estimator_.C, metrics.accuracy_score(YTests, svc.predict(XTests))

#	initial the parameter
args = init()

#	get all jpg and gt file path
imgPathList, txtPathList = getFilePath(args["dir"])

#	start hogging image, return hog in X and action in Y
X, Y = HoG(imgPathList, txtPathList, args["size"], args["bin"], args["cell"], args["block"])

#	using support vector clasifer to train and test the dataset
meanScore, bestPenaltyC, accuracy = SVC(X, Y, args["kFold"])

print("Kernel = linear\n" +
		"Using " + str(args["kFold"]) + "-fold cross validation" +
		", mean score for cross validation is " + str(meanScore) +
		"\nPenalty parameter C=" + str(bestPenaltyC) +
		", Accuracy score=" + str(accuracy) )

	


