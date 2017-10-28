import math 
import numpy as np
from numpy import *
from sympy import E
from PIL import Image
import pylab as plt

def hisEqualized(his, img):
	imgArr = np.array(img)
	temp = np.zeros((imgArr.shape[0], imgArr.shape[1]))
	for i in range(imgArr.shape[0]):
		for j in range(imgArr.shape[1]):
			sumOfPix = sum( his[ :imgArr[i, j] ])
			temp[i, j] = 255 * sumOfPix / (imgArr.shape[0] * imgArr.shape[1])
	equalizedImg = Image.fromarray(temp)
	return equalizedImg

def histogram(grayImg):
	imgArr = np.array(grayImg)
	his = np.zeros( (256) )
	for i in range(256):
		temp = where(imgArr == i)
		his[i] = temp[0].shape[0]
	return his

def main(file):
	I = Image.open(file)
	#I.show()
	his = histogram(I)
	equalizedImg = hisEqualized(his, I).convert('L')
	equalizedImg.show()
	
	fig = plt.figure()
	fig.add_subplot(221)
	plt.plot(his)
	plt.title('Original histogram')

	fig.add_subplot(222)
	plt.plot(histogram(equalizedImg))
	plt.title('New histogram')
	plt.show()
main("../ImageData/2_bw.jpg")
#main("../ImageData/135069.jpg")
#main("../ImageData/388016.jpg")