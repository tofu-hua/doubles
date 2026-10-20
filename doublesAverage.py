# -*- coding: utf-8 -*-
#  Usage:  python3 doublesAverage.py data.csv value (e.g. python3 doubles.py volley.csv 0)

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import copy
import matplotlib.pyplot as plt
import os
from sys import argv
from locale import atof, atoi


"""# config"""

kwarg = {'start_mean' : 1500, 'update_var':False,
         'file_name':None, 'A': 1/3, 'B': 0}
# A and B are parameters in Alg. 1 in Hua et al. (2023)

init_sig = [40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 140, 150, 160, 170, 175, 180, 185, 190, 195, 200, 205, 210, 220, 250, 260, 280, 290, 300, 320, 350, 360, 370, 380,390, 400]

K_value = [18, 20, 25, 30, 35, 40, 50, 60, 70, 80]

"""# input arguments
par: 0 0r 1, with 0 corresponding to prediction based on the ranking at the end of a given year; and 1 corresponding to both predicting and updating for the test data
"""

dataname = argv[1]
pa = argv[2]
par = atoi(pa) 

dataS = dataname.split('.')
s0 = dataS[0] 
print(s0)

if s0 == 'volley':
    names = ['circuit','year','date','gender','winner1_name','winner2_name','loser1_name','loser2_name']
    cutpoint = 2014
    
if s0 == 'tennis':
    names = ['date','surface','winner1_name','winner2_name','loser1_name','loser2_name']
    cutpoint = 2017
           
data = pd.read_csv(dataname, parse_dates= [0], header=0, names = names) 
data['date'] = pd.to_datetime(data['date'])
data.head()
print(len(data))


"""# split"""

# split by year
Train = data[data['date'].dt.year <= cutpoint]
Test = data[data['date'].dt.year > cutpoint]
Train.tail()


folder =  'Average'+ s0 +'_' + pa 
folderS = folder+'/'

print(folder)
print(folderS)

try:
  os.mkdir(folderS)
except:
  pass

os.chdir(folder)


# get 2 inputs(winners, losers)
# col_name = ['surface','winner_name','loser_name'] if config['single'] else ['surface','winner1_name','winner2_name','loser1_name','loser2_name']
col_name =  ['winner1_name','winner2_name','loser1_name','loser2_name']
winners_all,losers_all = data[col_name[:len(col_name)//2]], data[col_name[len(col_name)//2:]]

Train, Test = Train[col_name], Test[col_name]
winners,losers = Train[col_name[:len(col_name)//2]], Train[col_name[len(col_name)//2:]]
winners_val,losers_val = Test[col_name[:len(col_name)//2]], Test[col_name[len(col_name)//2:]]

Test.head()

"""## DESCRI"""

print('training size: %d'%(Train.shape[0]))
print('testing size: %d'%(Test.shape[0]))


"""# probability model """

def sigmoid(cur_mu_w, cur_mu_l):
  b = np.log(10)/400 
  x = b*(cur_mu_w-cur_mu_l)
  return 1/(1+np.exp(-x))

"""# Elo (conservative)"""

def elo(winners, losers, K, 
         start_mean = 1500, file_name = None,
         mode = 'train', prior_ratings=None):
  if file_name==None: file_name = 'elo_' + mode + '.txt'
  mode = mode.lower()
  if mode =='train': 
    prior_ratings = defaultdict(lambda: start_mean) 
  elif mode not in ['test','train']: 
    print(f'WARNING: Ur input of mode is {mode} and the current mode is "Train" ')

  with open(file_name,'w', encoding='utf-8') as f:
    str2 = 'game, current_winner1, current_winner2, current_loser1, current_loser2, winner_mean, loser_mean, phat'
    f.write(str2+'\n')
    player_double = list(zip(winners['winner1_name'],winners['winner2_name'],losers['loser1_name'],losers['loser2_name']))      
    count, tie, total, total_discrepancy = 0,0,0,0
    b = np.log(10)/400
    for game in np.arange(winners.shape[0]):
        current_players = player_double[game]
        team_i = str(set([f'{current_players[0]}', f'{current_players[1]}']))
        team_j = str(set([f'{current_players[2]}', f'{current_players[3]}']))
        for team in [team_i,team_j]:
          prior_ratings[team] if team not in prior_ratings else None
        # collect prior rating in player_rating        
        new_rating = prior_ratings.copy()
        # calculate team meam & variance
        winner_mean = new_rating[team_i]
        loser_mean = new_rating[team_j]
        # compare and count the correct prediction
        if winner_mean > loser_mean:
          count +=1
        elif winner_mean == loser_mean:
          tie +=1
        total +=1

        # calculate p_iq ,p_qi   
        p_iq = sigmoid(winner_mean, loser_mean)
        p_qi = 1-p_iq        
        # define s result
        winner_s, loser_s = 1, 0
        # Ki, Li
        K_i = K*(winner_s-p_iq)
        K_q = K*(loser_s-p_qi)
        # sum discrepancy
        discrepancy = -np.log(p_iq)
        total_discrepancy += discrepancy
        # update winner, loser
        new_rating[team_i] = winner_mean + K_i 
        new_rating[team_j] = loser_mean + K_q 
        # save record
        str2 ='%2d, %s, %s, %s, %s, %4.3f, %4.3f, %.4f\n'%(game+1,
                                                    current_players[0], current_players[1],
                                                    current_players[2],current_players[3],
                                                    winner_mean + K_i,  loser_mean + K_q, p_iq)
        f.write(str2) # save
        # update prior rating
        if par == 0:
           prior_ratings = new_rating if mode =='train' else prior_ratings
        else:
           prior_ratings = new_rating   
    f.close()
    error = total-count-tie
    # (n) - (correct (a) + ties (t)) = error (e)
    misclassified_rate = error/(total-tie)
    return prior_ratings, misclassified_rate, (total,count,tie), total_discrepancy/winners.shape[0]
    # total(n) - (accurate(a) + ties(t)) = error (e)


"""# Elo-D (doubles) """

def eloD(winners, losers, K, 
         start_mean = 1500, file_name = None,
         mode = 'train', prior_ratings=None):
    if file_name==None: file_name = 'eloD_' + mode + '.txt'
    if mode.lower() =='train':
      prior_ratings = defaultdict(lambda: start_mean) 
    elif mode.lower() not in ['test','train']:
      print(f'WARNING: Ur input of mode is {mode} and the current mode is "Train" ')

    with open(file_name,'w', encoding="utf-8") as f:
      str2 = 'game, current_winner1, current_winner2, current_loser1, current_loser2, winner_mean_old, loser_mean_old, winner_mean, loser_mean, winner1_mean_new, winner2_mean_new, loser1_mean_new, loser2_mean_new, phat'
      f.write(str2+'\n')
      player_double = list(zip(winners['winner1_name'],winners['winner2_name'],losers['loser1_name'],losers['loser2_name']))
      
      count,tie,total,total_discrepancy = 0,0,0,0
      b = np.log(10)/400
      for game in np.arange(winners.shape[0]):
        current_players = player_double[game]
        current_winner1, current_winner2 = current_players[0],current_players[1]
        current_loser1, current_loser2 = current_players[2],current_players[3]

        # collect prior rating in player_rating        
        new_rating = prior_ratings.copy()

        # each player mean & variance
        winner1_mean = new_rating[current_winner1]
        winner2_mean = new_rating[current_winner2]
        loser1_mean = new_rating[current_loser1]
        loser2_mean = new_rating[current_loser2]
        # calculate team meam & variance
        winner_mean = sum([winner1_mean,winner2_mean])/2
        loser_mean = sum([loser1_mean,loser2_mean])/2
        # compare and count the correct prediction
        if winner_mean > loser_mean:
          count +=1
        if winner_mean == loser_mean:
          tie +=1
        total +=1

        # calculate p_iq ,p_qi   
        p_iq = sigmoid(winner_mean, loser_mean)
        p_qi = 1-p_iq
        # define s: game outcome
        winner_s, loser_s = 1, 0
        # Ki
        K_i = K*(winner_s-p_iq)
        K_q = K*(loser_s-p_qi)
        
        # sum discrepancy
        discrepancy = -np.log(p_iq)
        total_discrepancy += discrepancy
        
        # Indiviual skill update
        # update winner
        winner1_mean_new = winner1_mean + K_i
        winner2_mean_new = winner2_mean + K_i

        new_rating[current_winner1] = winner1_mean_new 
        new_rating[current_winner2] = winner2_mean_new
        # update loser
        loser1_mean_new = loser1_mean + K_q
        loser2_mean_new = loser2_mean + K_q

        new_rating[current_loser1] = loser1_mean_new 
        new_rating[current_loser2] = loser2_mean_new
       # save record
        str2 ='%2d, %s, %s, %s, %s, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %4.3f, %.4f\n'%(game+1,current_players[0],current_players[1],current_players[2],current_players[3],winner_mean,loser_mean,winner_mean + K_i,loser_mean + K_q,winner1_mean_new, winner2_mean_new, loser1_mean_new, loser2_mean_new, p_iq)
        f.write(str2) # save
        # update prior rating
        if par == 0:
           prior_ratings = new_rating if mode =='train' else prior_ratings
        else:
           prior_ratings = new_rating  
    f.close()
    error = total-count-tie
    misclassified_rate = error/(total-tie)
    return prior_ratings, misclassified_rate, (total,count,tie), total_discrepancy/winners.shape[0]
    # total(n) - (accurate(a) + ties(t)) = error (e)


    

"""# train and test for elo and eloD, with various K values"""

file_name = 'errRate_elo_eloD.txt'
print(file_name)

all_errE_tr=[]
all_errE_te=[]
all_numE=[]
all_discE=[]

all_errED_tr=[]
all_errED_te=[]
all_numED=[]
all_discED=[]

with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s, %2s, %2s, %2s\n'%('K','errTrain','errTest',"(n,a,t)",'discrepancy'))
  for i in range(len(K_value)):
    estimate, err_tr,_,disc_tr = elo(winners,losers,K=K_value[i],start_mean = 1500,file_name=None,mode='train')
    _, err_te,numE,_ = elo(winners_val,losers_val, K=K_value[i],  start_mean = 1500, file_name=None, prior_ratings=estimate,mode='test')
    all_errE_tr.append(err_tr)
    all_errE_te.append(err_te) 
    all_numE.append(numE)
    all_discE.append(disc_tr)
    str_w = f'{K_value[i]:3d}, {err_tr:.4f},{err_te:.4f},{numE},{disc_tr:.4f}'
    f.write(str_w+"\n") # save

  f.write('%3s, %2s, %2s, %2s, %2s\n'%('K','errTrain','errTest',"(n,a,t)",'discrepancy'))
  for i in range(len(K_value)):
    estimate, err_tr,_,disc_tr = eloD(winners,losers,K=K_value[i],start_mean = 1500)
    _, err_te,numE,_ = eloD(winners_val,losers_val, K=K_value[i],  start_mean = 1500, prior_ratings=estimate,mode='test')
    all_errED_tr.append(err_tr)
    all_errED_te.append(err_te) 
    all_numED.append(numE)
    all_discED.append(disc_tr)
    str_w = f'{K_value[i]:3d}, {err_tr:.4f},{err_te:.4f},{numE},{disc_tr:.4f}'
    f.write(str_w+"\n") # save
    
  f.write("\n") 
  f.write('%3s, %2s, %4s, %2s, %3s, %2s, %4s, %2s\n'%('K','errTest','discrep','ties','K','errTest','discrep','ties'))
  for i in range(len(K_value)):
    str_w =  f'{K_value[i]:3d} & {all_errE_te[i]:.4f} & {all_discE[i]:.4f} & {all_numE[i][2]:3d} & & & {K_value[i]:3d} & {all_errED_te[i]:.4f} & {all_discED[i]:.4f} & {all_numED[i][2]:3d}\\\\'
    f.write(str_w+"\n") # save  
f.close()

""" for Elo, find K with min discrep in train data """
eloK = all_discE.index(min(all_discE))
print('(Elo) K with min discrepancy in train: %d'%(K_value[eloK]))

estimate, errtrain,_,_ = elo(winners,losers,K=K_value[eloK],start_mean = 1500) 
_, err2c,num2c,_ = elo(winners_val,losers_val, K=K_value[eloK],start_mean = 1500, prior_ratings=estimate,mode='test')


""" for Elo-D, find K with min discrep in train data """
eloK_D = all_discED.index(min(all_discED))
print('(Elo-D) K with min discrepancy in train: %d'%(K_value[eloK_D]))

estimate, errtrain,_,_ = eloD(winners,losers,K=K_value[eloK_D],start_mean = 1500) 
_, err2cD,num2cD,_ = eloD(winners_val,losers_val, K=K_value[eloK_D],start_mean = 1500,    prior_ratings=estimate,mode='test')

plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.plot(K_value, all_errE_te, color='b', label='Elo')
plt.plot(K_value, all_errED_te, color='r', linestyle = '--', label='Elo_D')
plt.xlabel('K',fontsize=14)
plt.ylabel('error rate',fontsize=14)
plt.legend()
# plt.title('Elo and Elo-D for various K')
# plt.axvline(init_sig[sigMin_idx2],c='red')
title2 = 'errPlot_Elo_EloD'
plt.savefig(title2)
# plt.show()
plt.close()
