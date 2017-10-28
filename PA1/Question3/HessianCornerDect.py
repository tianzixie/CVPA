import math 
import numpy as np
from numpy import *
from sympy import E
from numpy import linalg as LA
from PIL import Image
from PIL import ImageDraw


def saveImg(Img, fileName):
    Img.convert("RGB").save(fileName,"PNG")

def drawCircleOnImg(Img, cornerness):
	print("Drawing Circle")
	Img = Img.convert('RGB')
	draw = ImageDraw.Draw(Img)
	count = 0
	for i in range(cornerness.shape[0]):
		for j in range(cornerness.shape[1]):
			if cornerness[i, j] == 1:
				draw.ellipse( (j - 2, i - 2, j + 2, i + 2), outline ='red')
				count = count + 1
	print( "Total circle drawing : " + str(count) )
	return Img

def hessianMatrix(Ixx, Ixy, Iyx, Iyy, threshold):
	cornerArr = np.zeros((Ixx.shape[0], Ixx.shape[1]))
	(H, W) = Ixx.shape[0], Ixx.shape[1]
	count = 0
	for i in range(H):
		for j in range(W):
			temp = np.array(( [ Ixx[i, j], Ixy[i, j] ], [ Iyx[i, j], Iyy[i, j] ]))
			lambdaFeature, lambadVector = LA.eig(temp)
			if lambdaFeature[0] > threshold and lambdaFeature[1] > threshold:
				count = count + 1
				cornerArr[i][j] = 1
	print(count)
	return cornerArr
	
def derivative(ImgArr):
	print("Doing derivative")
	det = np.array([-1, 0, 1])
	detArr = np.zeros(ImgArr.shape)
	for i in range(ImgArr.shape[0]):
		for j in range(1, ImgArr.shape[1] - 1):
			tempImg = ImgArr[i, j - 1: j + 2]
			mul = np.multiply(tempImg, det)
			detArr[i, j] = sum(mul)
	return detArr
	
def gradient(ImgArr):
	gradientI = np.gradient(ImgArr)
	Ix = gradientI[1]
	Iy = gradientI[0]
	return Ix, Iy

def main(file, savingName, threshold):
	I = Image.open(file).convert('L')
	imgArr = np.array(I)
	draw = ImageDraw.Draw(I)

	Ix = derivative(imgArr)
	Iy = derivative(imgArr.T).T
	Ixx = derivative(Ix)
	Iyy = derivative(Iy.T).T
	Ixy = derivative(Ix.T).T
	Iyx = derivative(Iy)
	cornerArr = hessianMatrix(Ixx, Ixy, Iyx, Iyy, threshold)
	newI = drawCircleOnImg(I, cornerArr)
	saveImg(newI, savingName)
	

main("../ImageData/input1.png", "../SavingImage/input1.jpg", 120)
main("../ImageData/input2.png", "../SavingImage/input2.jpg", 120)
main("../ImageData/input3.png", "../SavingImage/input3.jpg", 120)

