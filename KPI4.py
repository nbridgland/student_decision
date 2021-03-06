#Generate KPI4 for each offer, for the first 3 days of the validity period
#KPI4 = number of trx for people who received offer over number of trx for control group

from __future__ import division
import numpy as np
import json
from datetime import datetime, timedelta

from population import Population

def KPI4(population):
	transcript_file_name = population.transcript_file_name
	population_file_name = 'data/population.json'
	#population = Population.from_json(population_file_name)

	transcript=[]

	with open(transcript_file_name, 'r') as transcript_file:
		for line_number, line in enumerate(transcript_file):
			text = line.strip()
			if text != '':
				transcript.append(json.loads(text))
           
	num_control=len(population.people)-len(TargetPeople)

	TargetPeople=[]

	for line in transcript:
		if line['event']=='offer received':
			TargetPeople.append(line['person'])
          
	offers=population.portfolio.values()

	start_date=[0 for i in range(0,len(offers))]
	offer_ids=[0 for i in range(0,len(offers))]

	for i in range(0,len(offers)):
		start_date[i]=offers[i].valid_from
    
	for i in range(0,len(offers)):
		offer_ids[i]=offers[i].id

	num_trx_targ=[0 for i in range(0,len(offers))]
	num_trx_cont=[0 for i in range(0,len(offers))]

	offer_groups=[[] for i in range(0,len(offers))]

	for line in transcript:
		if line['event']=='offer received':
			offer_groups[offer_ids.index(line['value']['offer id'])].append(line['person'])
        
	for line in transcript:
		if line['event']=='transaction':
			for j in range(0,len(offer_groups)):
				if line['time']>=start_date[j] and line['time']<start_date[j]+72:
					if line['person'] in offer_groups[j]:
						num_trx_targ[j]+=1
					if line['person'] not in TargetPeople:
						num_trx_cont[j]+=1
	
	KPI4=[]

	for i in range(0,len(offers)):
		KPI4.append((num_trx_targ[i]/len(offer_groups[i]))/(num_trx_cont[i]/num_control))

	return KPI4




        