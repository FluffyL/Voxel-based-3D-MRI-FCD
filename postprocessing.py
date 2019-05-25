import os
import sys
import gzip
import shutil
import torch
import nibabel as nib
import numpy as np
import PIL
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as pl
from torchvision import transforms

def img_show(img):
	for i in range(img.shape[0]):
		io.imshow(img[i,:,:],cmap='gray')
		print(i)
		io.show()

def calculate_threshold(img, label, num, flag):

	s_label = label.shape
	region = np.array(label == num)
	region_img = img*region
	# seems can't use mean = region_img.sum()/region.sum() for those region_img = 0
	mean = 0.0
	for i in range(s_label[0]):
		for j in range(s_label[1]):
			for k in range(s_label[2]):
				mean = mean + region_img[i,j,k]
	mean = mean/region.sum()

	sd = 0.0
	for i in range(s_label[0]):
		for j in range(s_label[1]):
			for k in range(s_label[2]):
				if label[i, j, k] == num:
					sd = sd + (region_img[i,j,k]-mean)**2
	sd = np.sqrt(sd/region.sum())
	
	if flag == 0:#lower threshold
		return mean + 1/2 * sd
	else:#upper threshold
		return mean - 1/2 * sd

def convolution3D(img, kernel_size):
	'''
	5*5*5 conv with all kernel value = 1
	'''
	simg = img.shape
	kernel = np.ones((kernel_size,kernel_size,kernel_size))

	conv_img = np.zeros((simg[0]-kernel_size+1,simg[1]-kernel_size+1,simg[2]-kernel_size+1))
	for i in range(simg[0]-kernel_size+1):
		for j in range(simg[1]-kernel_size+1):
			for k in range(simg[2]-kernel_size+1):
				for a in range(i-(kernel_size//2),i+(kernel_size//2)):
					for b in range(j-(kernel_size//2),j+(kernel_size//2)):
						for c in range(k-(kernel_size//2),k+(kernel_size//2)):
							conv_img[i][j][k] = conv_img[i][j][k] + img[a][b][c]
				
	return conv_img
	
	
'''
f1 = '/home/lly/Desktop/intern_Xidian/binary/S01.native.mri.hdr'
img_T1 = nib.load(f1)
affine = img_T1.affine
inputs_T1 = img_T1.get_data()
save_T1 = nib.Nifti1Image(inputs_T1, affine)
save_T1.to_filename('/home/lly/Desktop/intern_Xidian/binary/S01.nii.gz')
'''

f1 = '/home/lly/Desktop/intern_Xidian/postprocessing/S01.nii.gz'
fl = '/home/lly/Desktop/intern_Xidian/postprocessing/S01.label.nii.gz'
img_T1 = nib.load(f1)
img_label = nib.load(fl)
inputs_T1 = img_T1.get_data()
inputs_label = np.round(img_label.get_data())

#thresholding
lowth = calculate_threshold(inputs_T1, inputs_label,num = 2, flag = 0)#num = label of GM
upth = calculate_threshold(inputs_T1, inputs_label, num = 3, flag = 1)#num = label of WM

binary = np.multiply(np.array(inputs_T1>lowth),np.array(inputs_T1<upth))
mask = np.array(inputs_label >= 2)
binary = np.multiply(binary, mask)
binary = binary[:,:,:,0]*255
#img_show(binary)

#convolution
conv = convolution3D(binary,kernel_size = 5)
img_show(conv)
