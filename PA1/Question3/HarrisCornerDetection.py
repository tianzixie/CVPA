import math 
import numpy as np
from numpy import *
from sympy import E
from numpy import linalg as LA
from PIL import Image
from PIL import ImageDraw
from scipy import signal
import time
import xlsxwriter

def saveImg(Img, fileName):
    Img.convert("RGB").save(fileName,"PNG")

def writeValInExcel(array, alpha, sigma):
	#print("Writing all values from array to excel")
	workbook = xlsxwriter.Workbook(str(alpha) + '_' + str(sigma) + '_pic.xlsx')
	worksheet = workbook.add_worksheet()
	row = 0
	for col, data in enumerate(array):
		worksheet.write_column(row, col, data)
	workbook.close()

def drawCircleOnImg(Img, cornerness, threshold):
	#print("Drawing Circle")
	ImgRGB = Img.convert('RGB')
	draw = ImageDraw.Draw(ImgRGB)
	count = 0
	for i in range(cornerness.shape[0]):
		for j in range(cornerness.shape[1]):
			if cornerness[i, j] > threshold:
				draw.ellipse( (j - 2, i - 2, j + 2, i + 2), outline ='red')
				count = count + 1
	#print( "Total circle drawing : " + str(count) )
	ImgRGB.show()
	return ImgRGB

def harrisCornerDetectionEig(LxArr, LyArr, LxyArr, alpha):
	#print("Doing Harris Corner Detection Eig")
	(H, W) = LxArr.shape
	cornerness = np.zeros((H, W))
	tStart = time.time()
	for i in range(H):
		for j in range(W):
			A = LxArr[i, j]
			B = LyArr[i, j]
			C = LxyArr[i, j]
			temp = np.array(( [ A, C ], [ C, B ]))
			lambdaFeature, lambadVector = LA.eig(temp)
			det = lambdaFeature[0] * lambdaFeature[1]
			trace = lambdaFeature[0] + lambdaFeature[1]
			cornerness[i][j] = det - alpha * trace
	tEnd = time.time()
	print("Costing time from L1*L2 - alpha * (L1+L2) = " + str(tEnd - tStart) )
	return cornerness

	
def harrisCornerDetection(LxArr, LyArr, LxyArr, alpha):
	#print("Doing Harris Corner Detection")
	(H, W) = LxArr.shape
	cornerness = np.zeros((H, W))
	tStart = time.time()
	for i in range(H):
		for j in range(W):
			A = LxArr[i, j]
			B = LyArr[i, j]
			C = LxyArr[i, j]
			det = A*B - C**2
			trace = (A + B)**2
			cornerness[i][j] = det - alpha * trace
	tEnd = time.time()
	print("Costing time from Det(H2) - alpha * Tr(H2) = " + str(tEnd - tStart) )
	return cornerness

def smoothImg(arr, mask):
	#print("Doing smoothing")
	(imgH, imgW) = arr.shape
	newImgArr = np.zeros((imgH, imgW))
	maskSize = mask.shape[1]
	for i in range( imgH ):
		for j in range((int)(maskSize/2), imgW - (int)(maskSize/2) ):
			scaleImg = arr[i, j - (int)(maskSize / 2) : j + (int)(maskSize / 2) + 1]
			convolution = np.multiply(scaleImg, mask)
			newImgArr[i, j] = convolution.sum()
	return newImgArr

# 1D Kernel
def gaussianKernel(sigma, kSize):
    center = (kSize - 1) / 2
    s2 = 2.0 * (sigma**2)
    g = np.zeros( (kSize, 1) )
    for i in range(kSize):
        x2 = (i - center)**2
        g[i] = (1/ ( pow((2*math.pi), 0.5) * sigma ) ) * exp( -x2/s2 )
    return g.T / g.sum()

def derivative(ImgArr):
	#print("Doing derivative")
	det = np.array([-1, 0, 1])
	detArr = np.zeros(ImgArr.shape)
	for i in range(ImgArr.shape[0]):
		for j in range(1, ImgArr.shape[1] - 1):
			tempImg = ImgArr[i, j - 1: j + 2]
			mul = np.multiply(tempImg, det)
			detArr[i, j] = sum(mul)
	return detArr

def main(file, savingName, mask, alpha, sigma, threshold):
	Img = Image.open(file).convert('L')
	I = np.array(Img)
	Ix = derivative(I)
	Iy = derivative(I.T).T
	LxS = smoothImg(np.multiply(Ix, Ix), mask)
	LyS = smoothImg(np.multiply(Iy, Iy).T, mask).T
	Lxy = smoothImg(Ix*Iy, mask)
	cornerness = harrisCornerDetection(LxS, LyS, Lxy, alpha)
	newImg = drawCircleOnImg(Img, cornerness, threshold)
	saveImg(newImg, savingName)
# define sigma and size of mask
maskSize = 3
sigma = 1.5
alpha = 0.04

G = gaussianKernel(sigma, maskSize)
main("../ImageData/input1.png", "../SavingImage/HarrisInput1.jpg", G, alpha, sigma, 2*10**8)
main("../ImageData/input2.png", "../SavingImage/HarrisInput2.jpg", G, alpha, sigma, 5*10**8)
main("../ImageData/input3.png", "../SavingImage/HarrisInput3.jpg", G, alpha, sigma, 5*10**7)




