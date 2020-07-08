# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:32:55 2020

@author: ASUS
"""

import creaDizionarioScreen2biradAndScreen2label as s2l
#import flipAndResizeSingleImg as rf
import os
import cv2
import numpy as np



# = ("./INbreast.xls") 
#pathToFolderPatients = "../CroppedGroupedPatients/" #"../temp/" #"../CroppedGroupedPatients/" 

#RITORNA DIZIONARIO TUTTE LE IMMAGINI DI TUTTI I PAZIENTI : LABEL PER OGNI IMMAGINE
def allScreen2label(xlsPath, gpimagespath):
	idbirad2birad = s2l.createDictIdBirad2birad(xlsPath)
	#print("Dizionario idBirad:birad di ciascuno screen:")
	#print(idbirad2birad)
	#print()
	screen2birad = s2l.createDictScreen2birad(idbirad2birad,gpimagespath)
	#print("Dizionario screen:birad di ciascun paziente (con esattamente 4 screen):")
	#print(screen2birad)
	#print()
	screen2label = s2l.createDictScreen2label(screen2birad)
	return(screen2label)


#RITORNA UNA LISTA DI PERCORSI DELLE IMMAGINI DEI PAZIENTI E UNA LISTA DI LABEL DI OGNI IMMAGINE DI OGNI PAZIENTE
#input: percorso di tutte le immagini in GroupedPatients, dizionario
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
		#immagine = cv2.imread(img)
		one_hot_labels = [0,0,0,0] #implicitamente significa: if no cancer (lbl =0) --> [0,0,0,0]
		
		
		if lbl == 1 and "MG_L" in img: #if left scan is benign --> [0,1,0,0]
			one_hot_labels[1] = 1
		elif lbl == 2 and "MG_L" in img: #if left scan is malignant --> [1,0,0,0]
			one_hot_labels[0] = 1
		elif lbl == 1 and "MG_R" in img: #if right scan is benign --> [0,0,0,1]
			one_hot_labels[3] = 1
		elif lbl == 2 and "MG_R" in img: #if right scan is malignant --> [0,0,1,0]
			one_hot_labels[2] = 1
		
		one_hot_labels_arr = np.array(one_hot_labels)
		
		temp = img.split("/")
		nome_immagine = temp[-1]
		
		listaImmagini.append(nome_immagine) #img
		listaLabels.append(one_hot_labels_arr) #lbl
		
	return listaImmagini,listaLabels


#RITORNA UNA LISTA DI NOMI DELLE IMMAGINI DEI PAZIENTI E UNA LISTA DI LABEL (lista di 4 elementi) DI OGNI IMMAGINE DI OGNI PAZIENTE PER OGNI VIEW
#input: lista di percorsi di immagini (stringhe), lista di labels associate alla lista di percorsi di immagini
def suddividiInListeImmaginiElabelsPerView(listaPercorsiImmaginiDaSuddividere, listaLabels):
	
	listaLccImg = []
	listaRccImg = []
	listaLmloImg = []
	listaRmloImg = []
	
	listaLccLbl = []
	listaRccLbl = []
	listaLmloLbl = []
	listaRmloLbl = []
	
	

#	listaPercorsiImmaginiDaAnalizzare = []
#	
#	for root, dirs, files in os.walk(gpimagespath):
#		for file in files:
#			if file.endswith(".png"):
#				listaPercorsiImmaginiDaAnalizzare.append(os.path.join(root, file))

	for img, lbl in zip(listaPercorsiImmaginiDaSuddividere, listaLabels):
				
		#img = img.replace(os.sep,"/")
		#immagine = cv2.imread(img)
		#immagine = rf.flipAndResizeImg(img)
		
		if "L_CC" in img:
			listaLccImg.append(img) #immagine #nome_immagine
			listaLccLbl.append(lbl) #one_hot_labels_arr
			
		elif "R_CC" in img:
			listaRccImg.append(img) #immagine #nome_immagine
			listaRccLbl.append(lbl) #one_hot_labels_arr
			
		elif "L_ML" in img:
			listaLmloImg.append(img) #immagine #nome_immagine
			listaLmloLbl.append(lbl) #one_hot_labels_arr
			
		elif "R_ML" in img:
			listaRmloImg.append(img) #immagine #nome_immagine
			listaRmloLbl.append(lbl) #one_hot_labels_arr
			
	return listaLccImg, listaLccLbl, listaRccImg, listaRccLbl, listaLmloImg, listaLmloLbl, listaRmloImg, listaRmloLbl
						
'''
screen2label = allScreen2label(loc,pathToFolderPatients)
print("Dizionario screen:label di ciascun paziente (con esattamente 4 screen):")
print(screen2label)
	
listaImg, listaLbl = suddividiPercorsiImmaginiElabelsInListe(pathToFolderPatients, screen2label)
print(listaImg, listaLbl)
print()
#X,Y = suddividiInListeImmaginiElabelsPerView(pathToFolderPatients,screen2label)
#print(type(X))
#print(Y)
#
print("suddividi in liste immagini e labels per view:")
lcc_x, lcc_y, rcc_x, rcc_y, lmlo_x, lmlo_y, rmlo_x, rmlo_y = suddividiInListeImmaginiElabelsPerView(listaImg,listaLbl)
print("lcc_x:", lcc_x, "lcc_y:", lcc_y)
print()
print("rcc_x:",rcc_x,"rcc_y_", rcc_y)
print()
print("lmlo_x:",lmlo_x, "lmlo_y:",lmlo_y)
print()
print("rmlo_x:",rmlo_x, "rmlo_y:",rmlo_y)
print()
'''