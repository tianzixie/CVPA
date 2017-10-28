import math 
import numpy as np
from numpy import *
from sympy import E
from PIL import Image
import sys

def saveImg(Img, fileName):
    Img.convert("RGB").save(fileName,"PNG")

def isInRange(x, y, W, H):
    if x >= W:
        return False
    elif x < 0:
        return False
    elif y >= H:
        return False
    elif y < 0:
        return False
    else:
        return True

def recursive(x, y, Iht, tHigh):
	(W, H) = Iht.shape
	if Iht[x, y] >= tHigh:
		return 255
	Iht[x, y] = -1
	if isInRange(x - 1, y, W, H):
		if Iht[x - 1, y] > 0:
			if recursive(x - 1, y, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255

	if isInRange(x - 1, y + 1, W, H):
		if Iht[x - 1, y + 1] > 0:
			if recursive(x - 1, y + 1, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255
				
	if isInRange(x, y + 1, W, H):
		if Iht[x, y + 1] > 0:
			if recursive(x, y + 1, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255
				
	if isInRange(x + 1, y + 1, W, H):
		if Iht[x + 1, y + 1] > 0:
			if recursive(x + 1, y + 1, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255
				
	if isInRange(x + 1, y, W, H):
		if Iht[x + 1, y] > 0:
			if recursive(x + 1, y, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255
				
	if isInRange(x + 1, y - 1, W, H):
		if Iht[x + 1, y - 1] > 0:
			if recursive(x + 1, y - 1, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255
				
	if isInRange(x, y - 1, W, H):
		if Iht[x, y - 1] > 0:
			if recursive(x, y - 1, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255
				
	if isInRange(x - 1, y - 1, W, H):
		if Iht[x - 1, y - 1] > 0:
			if recursive(x - 1, y - 1, Iht, tHigh) >= tHigh:
				Iht[x, y] = 255
				return 255

	return 0
    
def hysteresisThresholding(INMS, tHigh, tLow):
	Iht = np.array(INMS)
	strong = where(Iht >= tHigh)
	Iht[strong] = 255
	drop = where(Iht <= tLow)
	Iht[drop] = 0
	candidate = where(Iht < tHigh)
	for i in range(candidate[0].shape[0]):
		(x, y) = candidate[0][i], candidate[1][i]
		if Iht[x, y] != 0 and Iht[x, y] != 255:
			Iht[x, y] = recursive(x, y, Iht, tHigh)
	return Iht

def NMS(deX, deY, magIArr):
	nmsIArr = np.array(magIArr)
	imgArrX, imgArrY = magIArr.shape
	gradientOrient = np.arctan2(deY, deX)

	for y in range(1, imgArrY - 1):
		for x in range(1, imgArrX - 1):
			for k in range(16):
				minVal = -math.pi + (1 / 8) * k
				maxVal = -math.pi + (1 / 8) * (k + 1)
				if minVal <= gradientOrient[x, y] <= maxVal:
					tempArr = 0
					if k == 0 or k == 7 or k == 8 or k == 15:
						# --
						tempArr = np.array(( magIArr[x - 1, y], magIArr[x, y], magIArr[x + 1, y] ))
					elif k == 1 or k == 2 or k == 9 or k == 10:
						# /
						tempArr = np.array(( magIArr[x - 1, y + 1], magIArr[x, y], magIArr[x + 1, y - 1] ))
					elif k == 3 or k == 4 or k == 11 or k == 12:
						# |
						tempArr = np.array(( magIArr[x, y - 1], magIArr[x, y], magIArr[x, y + 1] ))
					elif k == 5 or k == 6 or k == 13 or k == 14:
						# \
						tempArr = np.array(( magIArr[x - 1, y - 1], magIArr[x, y], magIArr[x + 1, y - 1] ))

					if argmax(tempArr) != 1:
						nmsIArr[x, y] = 0
					break
	return nmsIArr

def magnitude(deIxArr, deIyArr):
	magArr = np.zeros(deIxArr.shape)
	magArr = (deIxArr**2 + deIyArr**2)**(0.5)
	return magArr

def convolveY(imgArr, mask):
	maskSize = mask.shape[0]
	imgArrX, imgArrY = imgArr.shape
	newImgArr = np.zeros(imgArr.shape)
	for x in range(imgArrX):
		for y in range((int)(maskSize/2), imgArrY - (int)(maskSize/2)):
			partImgArr = imgArr[x, y - (int)(maskSize/2): y + (int)(maskSize/2) + 1]
			newImgArr[x, y] = sum(np.multiply(partImgArr, mask))
	return newImgArr

def convolveX(imgArr, mask):
	maskSize = mask.shape[0]
	imgArrX, imgArrY = imgArr.shape
	newImgArr = np.zeros(imgArr.shape)
	for y in range(imgArrY):
		for x in range( (int)(maskSize/2), imgArrX - (int)(maskSize/2)):
			partImgArr = imgArr[x - (int)(maskSize/2) : x + (int)(maskSize/2) + 1, y]
			newImgArr[x, y] = sum(np.multiply(partImgArr, mask))
	return newImgArr

def gaussianD(sigma, kSize):
	center = (kSize - 1) / 2
	s2 = 2.0 * (sigma**2)
	g = np.zeros((kSize))
	for i in range(kSize):
		x2 = (i - center)**2
		g[i] = (1/ ( pow((2*math.pi), 0.5) * sigma ) ) * exp( -x2/s2 ) * -1 * (i-center) / sigma**3
	return g / -g[0] * 2

def gaussian(sigma, kSize):
	center = (kSize - 1) / 2
	s2 = 2.0 * (sigma**2)
	g = np.zeros( (kSize) )
	for i in range(kSize):
		x2 = (i - center)**2
		g[i] = ( 1/ ( pow((2*math.pi), 0.5) * sigma ) ) * exp( -x2/s2 )
	return g / sum(g)

def main(file, savingName, G, Gd, sigma, maskSize):
	I = Image.open(file).convert('L')
	ImgArr = np.array(I).T
	sys.setrecursionlimit(ImgArr.shape[0]*ImgArr.shape[1])

	"""
	array = np.array( ( [1,1,1,1,1],
                        [1,1,1,1,1],
                        [1,1,2,3,1],
                        [1,2,1,3,1],
                        [1,1,1,1,1] ) )

	IxArr = convolveX(array, mask)
	print
	input("")

	"""
	IxArr = convolveX(ImgArr, G)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "_XConvolveGau.jpg"
	saveImg(Image.fromarray(IxArr.T), tempStr)
	IyArr = convolveY(ImgArr, G.T)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "_YConvolveGau.jpg"
	saveImg(Image.fromarray(IyArr.T), tempStr)

	#Image.fromarray(IxArr.T).show()
	#Image.fromarray(IyArr.T).show()

	deIxArr = convolveX(IxArr, Gd)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "_XConvolveGauD.jpg"
	saveImg(Image.fromarray(deIxArr.T), tempStr)
	deIyArr = convolveY(IyArr, Gd.T)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "_YConvolveGauD.jpg"
	saveImg(Image.fromarray(deIyArr.T), tempStr)
	#Image.fromarray(deIxArr.T).show()
	#Image.fromarray(deIyArr.T).show()
	
	magIArr = magnitude(deIxArr, deIyArr)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "_magnitude.jpg"
	saveImg(Image.fromarray(magIArr.T), tempStr)
	#Image.fromarray(magIArr.T).show()
	
	NMSIArr = NMS(deIxArr, deIyArr, magIArr)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "_NMS.jpg"
	saveImg(Image.fromarray(NMSIArr.T), tempStr)
	#Image.fromarray(NMSIArr.T).show()
	
	IhtArr = hysteresisThresholding(NMSIArr, 120, 80)
	tempStr = "../SavingImage/" + savingName + "_" + str(sigma) + "H120_L80_IhtArr.jpg"
	saveImg(Image.fromarray(IhtArr.T), tempStr)
	#Image.fromarray(IhtArr.T).show()
	
	
strArr = np.array(["../ImageData/2_bw.jpg", "../ImageData/119082.jpg", "../ImageData/135069.jpg"])
strSavingArr = np.array( ["2_bw", "119082", "135069"] )
for i in range(3):
	sigma = 0.5 * (i + 1)
	maskSize = 3
	G = gaussian(sigma, maskSize)
	Gd = gaussianD(sigma, maskSize)
	for j in range(3):
		main(strArr[j],strSavingArr[j], G, Gd, sigma, maskSize)


