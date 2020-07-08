# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:43:41 2020

@author: ASUS
"""

import os 
import xlrd 

#returns a dictionary in which the keys 
#are the ids of the images file (int), and the values
#are the corresponding birad value (string format)
def createDictIdBirad2birad(xlsPath):
	listaIdBirad = []
	listaBirad = []	
	
	wb = xlrd.open_workbook(xlsPath) 
	sheet = wb.sheet_by_index(0) 

	for i in range(1,sheet.nrows-2): #
		idBirad = int(sheet.cell_value(i, 3))
		listaIdBirad.append(idBirad)
		#print(idBirad)
		

	for i in range(1,sheet.nrows-2): #
		birad = str(sheet.cell_value(i, 5))
		listaBirad.append(birad)
		#print(birad)
		
	#dictionary int:string
	idBirad2birad = dict(zip(listaIdBirad,listaBirad))
	#print(idBirad2birad)
	return idBirad2birad


#returns a dictionary in which the keys are the .png file of each image (string format)
#and the values are the corresponding birad value (int)
def createDictScreen2birad(dizionarioId2birad, percorsoGroupedPatients):
	patQuattroScreenIdBirad2birad = dict()
	for foldername, subfolders, filenames in os.walk(percorsoGroupedPatients):
		for file in filenames:
				if file.endswith(".png"):
					patQuattroScreenIdBirad2birad[file] = None
	#print(patQuattroScreenIdBirad2birad)
	
	
	
	for idBirad in dizionarioId2birad:
		for patient in patQuattroScreenIdBirad2birad:
			if str(idBirad) in patient:
				patQuattroScreenIdBirad2birad[patient] = int(dizionarioId2birad[idBirad][0]) 
	#print(patQuattroScreenIdBirad2birad)
	return patQuattroScreenIdBirad2birad


#return a dictionary in which the keys are the .png file of each image (string format)
#and the values are the labels (int) (mapped from the birad value)
def createDictScreen2label(dizionarioScreen2birad):
	dictScreen2label = dict()
	for key, value in dizionarioScreen2birad.items():
		if value == 1:
			dictScreen2label[key] = 0
		elif value == 2 or value == 3:
			dictScreen2label[key] = 1
		else:
			dictScreen2label[key] =2	
	return dictScreen2label