# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:32:55 2020

@author: ASUS
"""

import creaDizionarioScreen2biradAndScreen2label as s2l
import os
import cv2
import numpy as np




#returns a dictionary in which the keys are the .png iamge files
#and the values are the corresponding labels
def allScreen2label(xlsPath, gpimagespath):
	idbirad2birad = s2l.createDictIdBirad2birad(xlsPath)

	screen2birad = s2l.createDictScreen2birad(idbirad2birad,gpimagespath)

	screen2label = s2l.createDictScreen2label(screen2birad)
	return(screen2label)


#returns 2 lists: the first list contains all the .png image files, while
#the second list contains the corresponding label transformed into an array of 4 elements (one hot label)
def suddividiPercorsiImmaginiElabelsInListe(gpimagespath, dizAllScreen2label):
	
	listaImmagini = []
	listaLabels = []
	listaPercorsiImmaginiDaAnalizzare = []
	
	for root, dirs, files in os.walk(gpimagespath):
		for file in files:
			if file.endswith(".png"):
				listaPercorsiImmaginiDaAnalizzare.append(os.path.join(root, file))
						
	for img, lbl in zip(listaPercorsiImmaginiDaAnalizzare, dizAllScreen2label.values()):
		img = img.replace(os.sep,"/")
		one_hot_labels = [0,0,0,0] #implicitely means: if no cancer (lbl =0) --> [0,0,0,0]
		
		
		if lbl == 1 and "MG_L" in img: #if left scan is benign --> [0,1,0,0]
			one_hot_labels[1] = 1
		if lbl == 2 and "MG_L" in img: #if left scan is malignant --> [1,0,0,0]
			one_hot_labels[0] = 1
		if lbl == 1 and "MG_R" in img: #if right scan is benign --> [0,0,0,1]
			one_hot_labels[3] = 1
		if lbl == 2 and "MG_R" in img: #if right scan is malignant --> [0,0,1,0]
			one_hot_labels[2] = 1
		
		one_hot_labels_arr = np.array(one_hot_labels)
		
		temp = img.split("/")
		nome_immagine = temp[2]
		
		listaImmagini.append(nome_immagine) #img
		listaLabels.append(one_hot_labels_arr) #lbl
		
	return listaImmagini,listaLabels


#for each view returns a list of .png image file and a list of one hot label corresponding to each view
def suddividiInListeImmaginiElabelsPerView(listaPercorsiImmaginiDaSuddividere, listaLabels):
	
	listaLccImg = []
	listaRccImg = []
	listaLmloImg = []
	listaRmloImg = []
	
	listaLccLbl = []
	listaRccLbl = []
	listaLmloLbl = []
	listaRmloLbl = []
	

	for img, lbl in zip(listaPercorsiImmaginiDaSuddividere, listaLabels):

		
		if "L_CC" in img:
			listaLccImg.append(img) 
			listaLccLbl.append(lbl)
			
		elif "R_CC" in img:
			listaRccImg.append(img)
			listaRccLbl.append(lbl) 
			
		elif "L_ML" in img:
			listaLmloImg.append(img) 
			listaLmloLbl.append(lbl) 
			
		elif "R_ML" in img:
			listaRmloImg.append(img) 
			listaRmloLbl.append(lbl) 
			
	return listaLccImg, listaLccLbl, listaRccImg, listaRccLbl, listaLmloImg, listaLmloLbl, listaRmloImg, listaRmloLbl