import math 
import numpy as np
from numpy import *
from sympy import E
from PIL import Image

def binaryImg(img, threshold):
	binaryTemp = np.array(img)
	aboveT = where(binaryTemp >= threshold)
	lowerT = where(binaryTemp < threshold)
	binaryTemp[aboveT] = 255
	binaryTemp[lowerT] = 0
	Ibin = Image.fromarray(binaryTemp)
	return Ibin

def entropy(prob):
    entropies = np.zeros( (256) )
    
    for i in range(255):
    	probA = 0
    	probB = 0
    	lnProbA = 0
    	lnProbB = 0
    	hA = 0
    	hB = 0

    	# A prob and ln prob
    	for j in range(i):
    		probA = probA + prob[j]
    		if prob[j] != 0:
    			lnProbA = lnProbA + (prob[j] * np.log(prob[j]))
    	# B prob and ln prob
    	for j in range(i, 256):
    		probB = probB + prob[j]
    		if prob[j] != 0:
    			lnProbB = lnProbB + (prob[j] * np.log(prob[j]))

    	if probA != 0:
    		hA = np.log(probA) - lnProbA / probA
    	else:
    		hA = 0

    	if probB != 0:
    		hB = np.log(probB) - lnProbB / probB
    	else:
    		hB = 0
    	entropies[i] = hA + hB

    return argmax(entropies)

def PDF(his):
	sumOfPix = sum(his)
	#print(sumOfPix)
	prob = np.zeros( (256) )
	for i in range(256):
		prob[i] = his[i] / sumOfPix
	return prob

def histogram(grayImg):
	imgArr = np.array(grayImg)
	his = np.zeros( (256) )
	for i in range(256):
		temp = where(imgArr == i)
		his[i] = temp[0].shape[0]
	return his

def main(file):
	I = Image.open(file)
	his = histogram(I)
	prob = PDF(his)
	threshold = entropy(prob)
	Ibin = binaryImg(I, threshold)
    
#main("../ImageData/119082.jpg")
#main("../ImageData/388016.jpg")
main("../ImageData/2_bw.jpg")




