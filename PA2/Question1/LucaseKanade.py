import math 
import numpy as np
import scipy.signal as si
import math
from numpy import *
from PIL import Image
from PIL import ImageDraw
from scipy import signal

def draw(u, v, img):
	img = img.convert('RGB')
	draw = ImageDraw.Draw(img)
	for i in range(0, u.shape[0],5):
		for j in range(0, u.shape[1],5):
			if (u[i][j]**2 + v[i][j]**2)**0.5 > 2:
				draw.line( (i, j) + (i+u[i][j]*2, j+v[i][j]*2) , fill=255)
	img.show()

def derivative(img1, img2):
	x = np.array(([-1, 1], [-1, 1]))
	y = np.array(([-1, -1], [1, 1]))
	fx = si.convolve2d(img1, x, mode='same') + si.convolve2d(img2, x, mode='same')
	fy = si.convolve2d(img1, y, mode='same') + si.convolve2d(img2, y, mode='same')
	ft = si.convolve2d(img1, np.ones((2,2)), mode='same') + si.convolve2d(img2, -1 * np.ones((2,2)), mode='same')
	return fx, fy, ft


def LucasKanade(img1, img2, windowSize):
	fx, fy, ft = derivative(img1, img2)
	x, y = fx.shape
	w = (int)(windowSize/2)
	u = np.zeros((x, y))
	v = np.zeros((x, y))
	for i in range(w, x - w + 1):
		for j in range(w, y - w + 1):
			Ix = fx[i - w:i + w, j - w:j + w].flatten(order='F')
			Iy = fy[i - w:i + w, j - w:j + w].flatten(order='F')
			It = -ft[i - w:i + w, j - w:j + w].flatten(order='F')
			A = np.array([Ix, Iy]).T
			u[i][j], v[i][j] =  np.dot( np.dot( np.linalg.inv( np.dot(A.T, A) ), A.T), It)
			if math.isnan(u[i][j]) == True:
				u[i][j] = 0
			if math.isnan(v[i][j]) == True:
				v[i][j] = 0
	return u, v, ft

def gaussianKernel(sigma, kSize):
    center = (kSize - 1) / 2
    s2 = 2.0 * (sigma**2)
    g = np.zeros( (kSize, 1) )
    for i in range(kSize):
        x2 = (i - center)**2
        g[i] = (1/ ( pow((2*math.pi), 0.5) * sigma ) ) * exp( -x2/s2 )
    return g.T / g.sum()

def zoomedOut(u, v):
	zoomedOutU = np.zeros( ( u.shape[0]*2, u.shape[1]*2 ) )
	zoomedOutV = np.zeros( ( v.shape[0]*2, v.shape[1]*2 ) )
	for i in range(0, zoomedOutU.shape[0], 2):
		for j in range(0, zoomedOutU.shape[1], 2):
			zoomedOutU[i][j] = u[ (int)(i / 2) ][ (int)(j / 2) ] 
			zoomedOutV[i][j] = v[ (int)(i / 2) ][ (int)(j / 2) ] 
			#print((zoomedOutU[i][j], zoomedOutV[i][j]))
			zoomedOutU[i+1][j] = zoomedOutU[i][j]
			zoomedOutU[i+1][j+1] = zoomedOutU[i][j]
			zoomedOutU[i][j+1] = zoomedOutU[i][j]

			zoomedOutV[i+1][j] = zoomedOutV[i][j]
			zoomedOutV[i+1][j+1] = zoomedOutV[i][j]
			zoomedOutV[i][j+1] = zoomedOutV[i][j]

	return zoomedOutU, zoomedOutV

def zoomedIn(img):
	smallScale = np.zeros(((int)(img.shape[0]/2), (int)(img.shape[1]/2)))
	for i in range(0, img.shape[0], 2):
		for j in range(0, img.shape[1], 2):
			smallScale[(int)(i / 2)][(int)(j / 2)] = img[i, j]	
	return smallScale

def main2(file1, file2, scaleTimes, g):

	img1 = Image.open(file1).convert('L')
	img2 = Image.open(file2).convert('L')
	imgArr1 = np.array(img1).T
	imgArr2 = np.array(img2).T
	smallScale1 = imgArr1
	smallScale2 = imgArr2

	for i in range(scaleTimes+1):
		smoothImg1 = si.convolve2d(smallScale1, g, mode='same')
		smoothImg2 = si.convolve2d(smallScale2, g, mode='same')
		smallScale1 = zoomedIn(smoothImg1)
		smallScale2 = zoomedIn(smoothImg2)

	u, v, ft = LucasKanade(smallScale1, smallScale2, 11)
	for i in range(scaleTimes+1):
		u, v = zoomedOut(u, v)
	draw(u, v, img2)
	

def main1(file1, file2):
	img1 = Image.open(file1).convert('L')
	img2 = Image.open(file2).convert('L')
	imgArr1 = np.array(img1).T
	imgArr2 = np.array(img2).T

	u, v, ft = LucasKanade(imgArr1, imgArr2, 21)
	draw(u, v, img2)
	
g = gaussianKernel(1.5, 5)
main2("../ImageData/basketball1.png", "../ImageData/basketball2.png", 0, g)
main2("../ImageData/grove1.png", "../ImageData/grove2.png", 0, g)
main1("../ImageData/basketball1.png", "../ImageData/basketball2.png")
main1("../ImageData/grove1.png", "../ImageData/grove2.png")

