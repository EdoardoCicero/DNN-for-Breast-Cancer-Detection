# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:09:33 2020

@author: ASUS
"""

import os
import pickle

dcm_images_path = r'.\INBreast_Dataset\AllDICOMs'
image_format = '.dcm' #'.png'

#CREA DIZIONARIO PAZIENTI:NUMERO_SCREENS_PER_PAZIENTE
def creaDictPazienti2screens(percorsoCartellaDcms):
	patient2screens= dict()
	listaDcms = os.listdir(percorsoCartellaDcms) #(vecchio my_list = os.listdir(pathToDcms) )
	for elem in listaDcms:
		if(image_format in elem): #".png"
			patient_id = elem[9:25]
			#print(patient_id)
			if(patient_id) in patient2screens:
				patient2screens[patient_id] = patient2screens[patient_id] +1
			else:
				patient2screens[patient_id] = 1		
	return patient2screens


#CREA UNA LISTA DI SOLI PAZIENTI CHE HANNO NUMERO DI SCREENS DIVERSE DA 4 (ex. RCC-LCC-LCC-RML-LML-LML)
def creaListaPazientiEsclusi(dizionarioPazienti2screens):
	listaPazientiEsclusi = []
	for key in list(dizionarioPazienti2screens.keys()):
		if(dizionarioPazienti2screens[key] != 4):
			#print(key,":",patient2screens[key])
			listaPazientiEsclusi.append(key)
			
	return listaPazientiEsclusi


patient2screens = creaDictPazienti2screens(dcm_images_path)
excluded = creaListaPazientiEsclusi(patient2screens)

exams = {}
for image_file in os.listdir(dcm_images_path):
	if image_file.endswith(image_format):
		#print(image_file)
		case = image_file.split('_')[1]
		side = image_file.split('_')[3]
		view = image_file.split('_')[4]
		#'_'.join([case,side,view])

		
		if case not in exams.keys():
			exams[case] = {'horizontal_flip': 'NO'} # aggiungere il nuovo paziente

		if case in excluded:
			continue
		
		if view == 'CC':
			if side == 'R':
				if not 'R-CC'in exams[case].keys():
					exams[case]['R-CC'] = [image_file[:-4]]
				else:
					excluded.append(case)
			elif side == 'L':
				if not 'L-CC'in exams[case].keys():
					exams[case]['L-CC'] = [image_file[:-4]]
				else:
					excluded.append(case)
		elif view == 'ML':
			if side == 'R':
				if not 'R-MLO'in exams[case].keys():
					exams[case]['R-MLO'] = [image_file[:-4]]
				else:
					excluded.append(case)
			elif side == 'L':
				if not 'L-MLO'in exams[case].keys():
					exams[case]['L-MLO'] = [image_file[:-4]]
				else:
					excluded.append(case)

				
            
# with open(dcm_images_path +'\\exam_list.pkl', 'wb') as f:
#     pickle.dump([exams[patient] for patient in exams if len(exams[patient])==5] , f)         



