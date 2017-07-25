#uses population from experiment.4-nocontrol.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

from population import Population

def add_demographics(population_file_name, transcript_file_name):
    population = Population.from_json(population_file_name)

    dt_fmt = '%Y%m%d'
    now = datetime.strptime('20170718', dt_fmt)


    #add ages, membership_ages, gender, to dataframe, from population file
    ages = list()
    for person in population.people.values():
        if person.dob is not None:
            age = now - datetime.strptime(person.dob, dt_fmt)
            if age.days/365.25 < 100:
                ages.append(age.days/365.25)
            else:
                age = 0
                ages.append(age)
        else:
            age = 0
            ages.append(age)

    membership_ages = list()
    for person in population.people.values():
        if person.became_member_on is not None:
            membership_age = now - datetime.strptime(person.became_member_on, dt_fmt)
            if membership_age.days/365.25 < 100:
                membership_ages.append(membership_age.days/365.25)
            else: membership_ages.append(0)
        else:
            membership_age = 0
            membership_ages.append(membership_age)

    genderlist = list()
    for person in population.people.values():
        if person.gender is not None:
            if person.gender == 'M':
                gender = [1,0,0]
                genderlist.append(gender)
            elif person.gender == 'F':
                gender = [0,1,0]
                genderlist.append(gender)
            elif person.gender == 'O':
                gender = [0,0,1]
                genderlist.append(gender)
        else:
            gender = [0,0,0]
            genderlist.append(gender)


    pop_df=pd.DataFrame(np.column_stack([population.people.keys(),ages,membership_ages, genderlist]),\
                    columns=["person_id","age","membership_age", "gender-m", "gender-f", "gender-o"])
    #pop_df.dropna(axis=0,how='any');
    pop_df=pop_df[~pop_df.isin(['na', 'nan']).any(axis=1)]

    pop_df["age"]=pop_df["age"].astype(float)
    pop_df["membership_age"]=pop_df["membership_age"].astype(float)
    pop_df["gender-m"]=pop_df["gender-m"].astype(float)
    pop_df["gender-f"]=pop_df["gender-f"].astype(float)
    pop_df["gender-o"]=pop_df["gender-o"].astype(float)

    #add_historicaldata
    stats = dict([(person_id, {'received': 0.0,
    #'group': 'control',
                           'viewed': 0.0,
                           'trx-before-offer': 0,
                           'spend-before-offer': 0.00,
                           'trx-after-offer': 0,
                           'spend-after-offer': 0.00,
                           'reward': 0}) for person_id in population.people])

    transcript_file_name = 'data/transcript.json'
    with open(transcript_file_name, 'r') as transcript_file:
        for line_number, line in enumerate(transcript_file):
            text = line.strip()
            if text != '':
                record = json.loads(text)

                if record['event'] == 'offer received':
                    stats[record['person']]['received'] += 1

                if record['event'] == 'offer viewed':
                    stats[record['person']]['viewed'] += 1

                if record['event'] == 'transaction':
                    if stats[record['person']]['viewed']>0:
                        stats[record['person']]['trx-after-offer'] += 1
                        stats[record['person']]['spend-after-offer'] += record['value']['amount']
                    else:
                        stats[record['person']]['trx-before-offer'] += 1
                        stats[record['person']]['spend-before-offer'] += record['value']['amount']

                if record['event'] == 'offer completed':
                    stats[record['person']]['reward']+=1

    # put dictionary stats and dictionary population.people together
    popnstat=[population.people,stats]
    popnstat=dict((k, [d[k] for d in popnstat]) for k in popnstat[0].keys());

    spending = list()
    liftspending=list()
    lifttransaction=list()
    transaction = list()
    viewing = list()
    viewedreceived = list()
    rewards = list()
    for idval in population.people.keys():
        received=popnstat[idval][1]['received']
        spend=popnstat[idval][1]['spend-before-offer']
        liftspend=popnstat[idval][1]['spend-after-offer']-popnstat[idval][1]['spend-before-offer']
        tranx=popnstat[idval][1]['trx-before-offer']
        lifttranx=popnstat[idval][1]['trx-after-offer']-popnstat[idval][1]['trx-before-offer']
        view=popnstat[idval][1]['viewed']
        #group=popnstat[idval][1]['group']
        reward=popnstat[idval][1]['reward']
        spending.append(spend)
        liftspending.append(liftspend)
        lifttransaction.append(lifttranx)
        transaction.append(tranx)
        viewing.append(view)
        if (received>0 and view > 0):
            viewedreceived.append(float(view)/received)
            rewards.append(float(reward))
        elif (received==0):
            viewedreceived.append(0.5)
            rewards.append(float(reward))
        else:
            viewedreceived.append(0)
            rewards.append(0.5)
    #make general dataframe
    pop_df=pd.DataFrame(np.column_stack([population.people.keys(),ages,membership_ages,genderlist,spending,transaction,viewedreceived, liftspending, lifttransaction, rewards]),\
                    columns=["person_id","age","membership_age","gender-m","gender-f", "gender-o", "spend", "trx", "viewedreceived", "liftspend", "lifttrx", "reward"])
    #pop_df.dropna(axis=0,how='any');
    pop_df=pop_df[~pop_df.isin(['na', 'nan']).any(axis=1)]

    pop_df["age"]=pop_df["age"].astype(float)
    pop_df["membership_age"]=pop_df["membership_age"].astype(float)
    pop_df["spend"]=pop_df["spend"].astype(float)
    pop_df["trx"]=pop_df["trx"].astype(float)
    pop_df["viewedreceived"]=pop_df["viewedreceived"].astype(float)
    pop_df["reward"]=pop_df["reward"].astype(float)
    return pop_df

def segment_pop(population_file_name, transcript_file_name, n):
    pop_df=add_demographics(population_file_name, transcript_file_name)
    cluster = KMeans(n_clusters=n)
    pop_df['cluster'] = cluster.fit_predict(pop_df[pop_df.columns[2:12]])
    pop_df['cluster'] = pop_df['cluster'].astype(int)

    pop_df['cluster'] = pop_df['cluster'].astype("int")
    pop_clustered=pop_df[["person_id","cluster"]].set_index('person_id').T.to_dict('list')
    with open('./data/population_clustered.json', 'w') as outfile:
        json.dump(pop_clustered, outfile)
