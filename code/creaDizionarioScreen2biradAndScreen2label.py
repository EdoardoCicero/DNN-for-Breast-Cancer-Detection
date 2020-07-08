# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:43:41 2020

@author: ASUS
"""

import os 
import xlrd 


def createDictIdBirad2birad(xlsPath):
	listaIdBirad = []
	listaBirad = []	
	
	wb = xlrd.open_workbook(xlsPath) 
	sheet = wb.sheet_by_index(0) 

	#per prendere l'idBirad di ogni seno di ogni paziente
	#scorro la 4 colonna del file xls, dalla seconda riga all'ultima riga -2 (per prendere solo i numeri che corrispondono all'idBirad di ogni paziente)
	for i in range(1,sheet.nrows-2): #
		idBirad = int(sheet.cell_value(i, 3))
		listaIdBirad.append(idBirad)
		#print(idBirad)
		
	#per prendere il valore birad di ciascun idBirad di ogni seno di ogni paziente
	#scorro la quinta colonna senza prendere il considerazione il primo elemento (l'header)
	#e gli ultimi due (celle (=stringhe) vuote nell'xls)
	for i in range(1,sheet.nrows-2): #
		birad = str(sheet.cell_value(i, 5))
		listaBirad.append(birad)
		#print(birad)
		
	#dizionario int:string
	idBirad2birad = dict(zip(listaIdBirad,listaBirad))
	#print(idBirad2birad)
	return idBirad2birad



def createDictScreen2birad(dizionarioId2birad, percorsoGroupedPatients):
	#ASSUMPTION: LE SCREEN PRESE IN CONSIDERAZIONE SONO DEI PAZIENTI CHE HANNO ESATTAMENTE 4 SCREENS
	#inizializzo dizionario con chiavi=screens e value=None
	patQuattroScreenIdBirad2birad = dict()
	for foldername, subfolders, filenames in os.walk(percorsoGroupedPatients):
		for file in filenames:
				if file.endswith(".png"): #così non prende in considerazione il metadata.pkl
					patQuattroScreenIdBirad2birad[file] = None
	#print(patQuattroScreenIdBirad2birad)
	
	
	
	#creo dizionario con chiavi=screens e value=biradDiOgniScreen
	for idBirad in dizionarioId2birad:
		for patient in patQuattroScreenIdBirad2birad:
			if str(idBirad) in patient:
				#trasforma da intero a stringa e prende in considerazione solo la prima cifra 
				#(cosìcchè ad es. il birad "4c"(string) diventa 4(int))
				#è un dizionario string:int
				patQuattroScreenIdBirad2birad[patient] = int(dizionarioId2birad[idBirad][0]) 
	#print(patQuattroScreenIdBirad2birad)
	return patQuattroScreenIdBirad2birad


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




def main(xls_path, folder_path):
    
    # xls_path =
    # loc = (r"E:\Edoardo\Universita\Magistrale\Neural_Networks\Project\INBreast_Dataset\INbreast.xls") 
    # pathToFolderPatients = r"E:\Edoardo\Universita\Magistrale\Neural_Networks\Project\INBreast_Dataset\AllDICOMs_cropped"
    
    idbirad2birad = createDictIdBirad2birad(xls_path)
    # print("Dizionario idBirad:birad di ciascuno screen:")
    #print(idbirad2birad)
    # print()
    screen2birad = createDictScreen2birad(idbirad2birad,folder_path)
    # print("Dizionario screen:birad di ciascun paziente (con esattamente 4 screen):")
    # print(screen2birad)
    # print()
    screen2label = createDictScreen2label(screen2birad)
    # print("Dizionario screen:label di ciascun paziente (con esattamente 4 screen):")
    #print(screen2label)
    # print()
    
    
    c1 =c2=c3=c4=c5=c6 = 0
    for i in screen2birad.keys():
    	birad = screen2birad[i]
    	if birad == 1:
    		c1=c1+1
    	elif birad == 2:
    		c2=c2+1
    	elif birad == 3:
    		c3=c3+1
    	elif birad == 4:
    		c4=c4+1
    	elif birad == 5:
    		c5=c5+1
    	else:
    		c6=c6+1
    print("screen label 0:",c1)
    print("screen label 1:",c2+c3)
    print("screen label 2:",c4+c5+c6)


    return screen2birad



#main (r"E:\Edoardo\Universita\Magistrale\Neural_Networks\Project\INBreast_Dataset\INbreast.xls", r"E:\Edoardo\Universita\Magistrale\Neural_Networks\Project\INBreast_Dataset\AllDICOMs_cropped")


