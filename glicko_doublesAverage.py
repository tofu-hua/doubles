# -*- coding: utf-8 -*-
# Usage: python3 glicko_doublesAverage.py volley.csv value1 value2
#        (e.g. python3 glicko_doublesAverage.py volley.csv 0 80)
#       where "value1" is a number that controls the way of prediction --
#                 value1 = 0:
#              predict test data based on rating at the end of train data
#                 value1 != 0:
#              predict test data and keep updating rating after each game
#       "value2" is the number of lower bound on sigma

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import copy
import matplotlib.pyplot as plt
from datetime import *
from scipy.optimize import minimize
import os
from sys import argv
from locale import atof, atoi


"""# config"""

kwarg = {'start_mean' : 1500, 'update_var':True}

# set start_sig, mode
start_period = 10
start_sig = 50
start_nu = 15
init_sig = [30, 40, 50, 60, 80, 100, 120, 140, 150, 160, 180, 200, 220, 250, 260, 300, 320, 360, 380, 400]
# init_sig = [30, 40, 50, 80, 100, 120, 140, 150, 160, 180, 200, 220, 250, 260, 300]
nu = [0, 5, 8, 10, 15]
# nu = [0, 5, 8, 10, 15, 20, 25, 30]
period = [1, 7]
# period = [1, 7, 10]

"""# data file """

dataname = argv[1]
pa = argv[2]
par = atoi(pa)
pa1 = argv[3]
par1 = atoi(pa1)

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


"""# output folder """

folder =  'AVEglickoOUTPUT'+ s0 +'_' + pa + '_' + pa1
folderS = folder+'/'

print(folder)
print(folderS)

try:
  os.mkdir(folderS)
except:
  pass

os.chdir(folder)


"""# split to train and test"""

# split by year
Train = data[data['date'].dt.year <= cutpoint]
Test = data[data['date'].dt.year > cutpoint]
Train.tail()

winners,losers = Train[['winner1_name','winner2_name']], Train[['loser1_name','loser2_name']]
winners_val,losers_val = Test[['winner1_name','winner2_name']], Test[['loser1_name','loser2_name']]
date_time = Train['date']
date_time_val = Test['date']


"""# probability model """

def sigmoid(cur_mu_w, cur_mu_l):
  b = np.log(10)/400 
  x = b*(cur_mu_w-cur_mu_l)
  return 1/(1+np.exp(-x))

def g_function(q, sigmasq):
  return(1/np.sqrt((1+3*(q**2)*(sigmasq)/(np.pi**2))))



"""# glicko (conservative)"""

def glicko(game_dates, winners, losers, start_sig, days, c, start_mean = 1500, update_var=True, file_name = None, mode='train', prior_ratings=None): 
    if mode.lower() =='train':
        prior_ratings = defaultdict(lambda: (start_mean, start_sig**2))
    if file_name==None: file_name = 'glicko_'+ mode
    elif mode.lower() not in ['test','train']: 
        print(f'WARNING: Ur input of mode is {mode} and the current mode is "Train" ')
    
    with open(file_name, 'w', encoding="utf-8") as f:
        str2 = 'game, winner_mean_old, loser_mean_old, winner_1, winner_2, loser_1, loser_2, phat'
        f.write(str2+'\n')
        count, tie, total, total_discrepancy = 0, 0, 0, 0
        q = np.log(10)/400
        terms = int(np.ceil((max(game_dates) - min(game_dates) + timedelta(days=1))/timedelta(days = days)))
        time_stamp = min(game_dates)
        game_stamp = 0
        for i in range(terms):        
            tourn_stamp = (game_dates >= time_stamp) & (game_dates < time_stamp + timedelta(days = days))
            tourn_winners = winners[tourn_stamp]
            tourn_winners = tourn_winners.reset_index(drop=True)
            tourn_losers = losers[tourn_stamp]
            tourn_losers = tourn_losers.reset_index(drop=True)
            tourn_teams = []
            new_rating = prior_ratings.copy()
            update_amount = defaultdict(lambda: (0, 0)) #(mean_update_amount, delta_squ_amount)
            # print(i, tourn_winners.shape[0], str(time_stamp))
            
            for game in np.arange(tourn_winners.shape[0]):
                game_stamp += 1
                cur_winner_1 = tourn_winners['winner1_name'][game]
                cur_winner_2 = tourn_winners['winner2_name'][game]
                cur_loser_1 = tourn_losers['loser1_name'][game]
                cur_loser_2 = tourn_losers['loser2_name'][game]
                current_players = [cur_winner_1, cur_winner_2, cur_loser_1, cur_loser_2]
                team_i = str(set(current_players[:2]))
                team_j = str(set(current_players[2:4]))
                for team in [team_i, team_j]:
                    prior_ratings[team] if team not in prior_ratings else None
                    tourn_teams.append(team) if team not in tourn_teams else None

                # calculate team mean & variance
                winner_mean, winner_var = prior_ratings[team_i]
                loser_mean, loser_var = prior_ratings[team_j]

                #accuracy
                total += 1
                if winner_mean>loser_mean:
                    count += 1
                elif winner_mean == loser_mean:
                    tie += 1
                    
                #calculate g_ij, g_winner, g_lower
                g_ij = g_function(q, winner_var + loser_var)
                g_winner = g_function(q, loser_var)
                g_loser = g_function(q, winner_var)
                
                #calculate p_ij, E(s|mu, mu_j, sigma_j^2)
                p_ij = 1/(1+10**(-g_ij*(winner_mean-loser_mean)/400))
                p_w = 1/(1+10**(-g_winner*(winner_mean-loser_mean)/400))
                p_l = 1/(1+10**(-g_loser*(loser_mean-winner_mean)/400))
                
                #calculate winners and losers sigma^2 in this game
                update_amount[team_i] = (update_amount[team_i][0] + g_winner*(1-p_w), update_amount[team_i][1]+(g_winner**2)*p_w*(1-p_w))
                update_amount[team_j] = (update_amount[team_j][0] + g_loser*(-p_l), update_amount[team_j][1]+(g_loser**2)*p_l*(1-p_l))

                #calculate total discrepancy
                discrepancy = -1 * np.log(p_ij)
                total_discrepancy += discrepancy
                
                #record
                str2 = '%2d, %4.3f, %4.3f, %s, %s, %.4f\n'%(game_stamp, winner_mean, loser_mean, team_i, team_j, p_ij)
                f.write(str2)


        
            #update and record team ratings
            for team in tourn_teams:
                sigmasq_new  = 1/(1/(new_rating[team][1]) + (q**2)*update_amount[team][1]) if update_var else new_rating[team][1] 
                new_rating[team] = (new_rating[team][0]+ q*sigmasq_new*update_amount[team][0], max(par1**2,sigmasq_new))
                #str2 = '%s, %s, %4.2f, %4.2f, %4.2f, %4.2f\n'%(game_stamp, team, prior_ratings[team][0], prior_ratings[team][1], new_rating[team][0],  new_rating[team][1])
                
                #f.write(str2) 
        
            #update the variance of the all_players
                ## f.write('\n')
                ## str2 = 'term, team,  rating, var'
                ## f.write(str2+'\n')
            for team in new_rating.keys():
                # new_rating[team] = (new_rating[team][0] , new_rating[team][1] + c**2)
                new_rating[team] = (new_rating[team][0] , min(new_rating[team][1] + c**2,start_sig**2))
                # new_rating[team] = (new_rating[team][0] , max(40**2, min(new_rating[team][1] + c**2,start_sig**2)))
                
                ##  str2 = '%d, %s, %4.2f, %4.2f\n'%(i, team, new_rating[team][0], new_rating[team][1])
                ##  f.write(str2)
                ## f.write('\n')

            #update prior_rating
            if par == 0:
               prior_ratings = new_rating if mode =='train' else prior_ratings
            else:
               prior_ratings = new_rating

            time_stamp = time_stamp + timedelta(days = days)

    f.close()
    error = total-count-tie
    misclassified_rate = error/(total-tie)
    return prior_ratings, misclassified_rate, (total,count,tie), total_discrepancy/winners.shape[0] 

estimate2c,err2ctrain,_,discrepancy = glicko(date_time, winners, losers, start_sig = start_sig, days = start_period, c = start_nu, **kwarg, mode='train', prior_ratings=None)

_,err2ctest,_,discrep = glicko(date_time_val, winners_val, losers_val, start_sig = start_sig, days = start_period, c = start_nu, **kwarg, mode='test', prior_ratings = estimate2c)




"""### glicko-D"""

def glickoD(game_dates, winners, losers, start_sig, days, c, start_mean = 1500, update_var=True, file_name=None, mode='train', prior_ratings=None):
  if mode.lower() =='train':
    prior_ratings = defaultdict(lambda: (start_mean, start_sig**2))
  elif mode.lower() not in ['test','train']: 
    print(f'WARNING: Ur input of mode is {mode} and the current mode is "Train" ')
  if file_name==None: file_name = 'glickoD_'+ mode
  with open(file_name, 'w', encoding="utf-8") as f:
    str2 = 'game, winner_mean_old, loser_mean_old, current_winner1, current_winner2, current_loser1, current_loser2, winner1_mean, winner2_mean, loser1_mean, loser2_mean, phat'
    f.write(str2+'\n')
    count, tie, total, total_discrepancy = 0, 0, 0, 0
    q = np.log(10)/400
    terms = int(np.ceil((max(game_dates) - min(game_dates) + timedelta(days=1))/timedelta(days = days)))
    time_stamp = min(game_dates)
    last_tourn_players = []
    game_stamp = 0
    for i in range(terms):
      tourn_stamp = (game_dates >= time_stamp) & (game_dates < time_stamp + timedelta(days = days))
      tourn_winners = winners[tourn_stamp]
      tourn_winners = tourn_winners.reset_index(drop=True)
      tourn_losers = losers[tourn_stamp]
      tourn_losers = tourn_losers.reset_index(drop=True)
      tourn_players = []
      new_rating = prior_ratings.copy()
      teammate_dict = dict()
      update_amount = defaultdict(lambda: (0, 0)) #(mean_update_amount, delta_squ_amount)
      for game in np.arange(tourn_winners.shape[0]):
        game_stamp += 1
        cur_winner_1 = tourn_winners['winner1_name'][game]
        cur_winner_2 = tourn_winners['winner2_name'][game]
        cur_loser_1 = tourn_losers['loser1_name'][game]
        cur_loser_2 = tourn_losers['loser2_name'][game]
        #record the teammates
        teammate_dict[cur_winner_1] = cur_winner_2
        teammate_dict[cur_winner_2] = cur_winner_1
        teammate_dict[cur_loser_1] = cur_loser_1
        teammate_dict[cur_loser_2] = cur_loser_2
        current_players = [cur_winner_1, cur_winner_2, cur_loser_1, cur_loser_2]
        for player in current_players:
          prior_ratings[player] if player not in prior_ratings else None
          tourn_players.append(player) if player not in tourn_players else None

        # calculate team mean & variance
        winner_1_mean, winner_1_var = prior_ratings[cur_winner_1]
        winner_2_mean, winner_2_var = prior_ratings[cur_winner_2]
        loser_1_mean, loser_1_var = prior_ratings[cur_loser_1]
        loser_2_mean, loser_2_var = prior_ratings[cur_loser_2]
        #team1 and team2
        mu_w = float(winner_1_mean + winner_2_mean)/2
        mu_l = float(loser_1_mean + loser_2_mean)/2
        var_w = float(winner_1_var + winner_2_var)/4
        var_l = float(loser_1_var + loser_2_var)/4

        #accuracy
        total += 1
        if mu_w > mu_l:
            count += 1
        elif mu_w == mu_l:
            tie += 1
        
        #calculate g_ij, g_winner, g_lower
        g_ij = g_function(q, var_w + var_l)
        g_winner = g_function(q, var_l)
        g_loser = g_function(q, var_w)
        
        #calculate p_ij, E(s|mu, mu_j, sigma_j^2)
        p_ij = 1/(1+10**(-g_ij*(mu_w - mu_l)/400))
        p_w = 1/(1+10**(-g_winner*(mu_w - mu_l)/400))
        p_l = 1/(1+10**(-g_loser*(mu_l - mu_w)/400))
        
        #calculate winners and losers sigma^2 in this game
        update_amount[cur_winner_1] = (update_amount[cur_winner_1][0] + g_winner*(1-p_w), update_amount[cur_winner_1][1]+(g_winner**2)*p_w*(1-p_w))
        update_amount[cur_winner_2] = (update_amount[cur_winner_2][0] + g_winner*(1-p_w), update_amount[cur_winner_2][1]+(g_winner**2)*p_w*(1-p_w))
        update_amount[cur_loser_1] = (update_amount[cur_loser_1][0] + g_loser*(-p_l), update_amount[cur_loser_1][1]+(g_loser**2)*p_l*(1-p_l))
        update_amount[cur_loser_2] = (update_amount[cur_loser_2][0] + g_loser*(-p_l), update_amount[cur_loser_2][1]+(g_loser**2)*p_l*(1-p_l))

        #calculate total discrepancy
        discrepancy = -1 * np.log(p_ij)
        total_discrepancy += discrepancy

        #record
        str2 = '%2d, %4.3f,%4.3f, %s, %s,%s, %s, %4.3f, %4.3f, %4.3f, %4.3f, %.4f\n'%(game_stamp, mu_w, mu_l, cur_winner_1, cur_winner_2, cur_loser_1, cur_loser_2, winner_1_mean, winner_2_mean, loser_1_mean, loser_2_mean, p_ij)
        f.write(str2)
        
      #update team ratings
      for player in tourn_players:
        teammate = teammate_dict[player]
        teammate_var = prior_ratings[teammate][1]
        team_var = (teammate_var + prior_ratings[player][1])/4
        player_var = prior_ratings[player][1]
        
        d2 = 1/((q**2)*update_amount[player][1])
        delta_it = team_var/(team_var + d2)
        new_rating[player] = (new_rating[player][0]+ ((1/2)*player_var/team_var)*(q/(1/(team_var)+(q**2)*update_amount[player][1]))*update_amount[player][0], max(par1**2,player_var*(1-(1/4)*(player_var/team_var)*delta_it)) if update_var else player_var)
        
        #str2 = '%s, %s, %4.3f, %4.3f\n'%(str(time_stamp), player, new_rating[player][0],  new_rating[player][1])
        #f.write(str2) 
        
      #update the variance of the nonplayed_players
      for player in new_rating.keys():
        new_rating[player] = (new_rating[player][0] , new_rating[player][1] + c**2)
        # new_rating[player] = (new_rating[player][0] , min(new_rating[player][1] + c**2, start_sig**2))
        # new_rating[player] = (new_rating[player][0] , max(40**2, min(new_rating[player][1] + c**2, start_sig**2)))
      
      #update prior_rating
      if par == 0:
         prior_ratings = new_rating if mode =='train' else prior_ratings
      else:
         prior_ratings = new_rating

      time_stamp = time_stamp + timedelta(days = days)

    f.close()
    error = total-count-tie
    misclassified_rate = error/(total-tie)
    return prior_ratings, misclassified_rate, (total,count,tie), total_discrepancy/winners.shape[0] 
    

estimate2,err2train,_,discrepancy = glickoD(date_time, winners, losers, start_sig = start_sig, days = start_period, c = start_nu, **kwarg, mode='train', prior_ratings=None)

_,err2test,_,discrep = glickoD(date_time_val, winners_val, losers_val, start_sig = start_sig, days = start_period, c = start_nu, **kwarg, mode='test', prior_ratings = estimate2)


"""# train and test for various sigma, c and day, update both mean and variance"""

d1 = len(period)
d2 = len(init_sig)
d3 = len(nu)

# save results by glicko
file_name = 'errRate_glicko'

all_err2ctrain = np.zeros((d1,d2,d3))
all_err2c = np.zeros((d1,d2,d3))
all_num2c = np.zeros((d1,d2,d3))
all_discrepancy2c = np.zeros((d1,d2,d3))
   
with open(file_name, 'w', encoding="utf-8") as f:
 for i in range(len(period)):
       f.write('%3s  %3d\n'%('period = ', period[i]))
       for j in range(len(init_sig)):
           f.write('%3s  %3d\n'%('sig = ', init_sig[j]))
           f.write('%3s, %2s, %2s, %2s, %2s\n'%('c','errTrain','errTest',"(n,a,t)",'discrepancy'))
           for k in range(len(nu)):
               estimate2c, err2ctrain,_,discrepancy2c = glicko(date_time, winners, losers,
                         start_sig=init_sig[j], days=period[i], c=nu[k], **kwarg, mode='train', prior_ratings=None)
               _, err2c,num2c,_ = glicko(date_time_val, winners_val, losers_val,
                         start_sig=init_sig[j], days=period[i], c=nu[k], **kwarg, mode='test', prior_ratings = estimate2c)
   
               all_err2ctrain[i,j,k] = err2ctrain
               all_err2c[i,j,k] = err2c
               all_num2c[i,j,k] = num2c[2]
               all_discrepancy2c[i,j,k] = discrepancy2c
               str_w = f'{nu[k]:3d}, {err2ctrain:.4f},{err2c:.4f},{num2c},{discrepancy2c:.4f}'
               f.write(str_w+"\n") # save
       
       f.write('\n')        
f.close()



# save results by glicko-D
file_name = 'errRate_glickoD'

all_err2train = np.zeros((d1,d2,d3))
all_err2 = np.zeros((d1,d2,d3))
all_num2 = np.zeros((d1,d2,d3))
all_discrepancy2 = np.zeros((d1,d2,d3))
   
with open(file_name, 'w', encoding="utf-8") as f:
 for i in range(len(period)):
       f.write('%3s  %3d\n'%('period = ', period[i]))
       for j in range(len(init_sig)):
           f.write('%3s  %3d\n'%('sig = ', init_sig[j]))
           f.write('%3s, %2s, %2s, %2s, %2s\n'%('c','errTrain','errTest',"(n,a,t)",'discrepancy'))
           for k in range(len(nu)):
               estimate2, err2train,_,discrepancy2 = glickoD(date_time, winners, losers,
                         start_sig=init_sig[j], days=period[i], c=nu[k], **kwarg, mode='train', prior_ratings=None)
               _, err2,num2,_ = glickoD(date_time_val, winners_val, losers_val,
                         start_sig=init_sig[j], days=period[i], c=nu[k], **kwarg, mode='test', prior_ratings = estimate2)
   
               all_err2train[i,j,k] = err2train
               all_err2[i,j,k] = err2
               all_num2[i,j,k] = num2[2]
               all_discrepancy2[i,j,k] = discrepancy2
               str_w = f'{nu[k]:3d}, {err2train:.4f},{err2:.4f},{num2},{discrepancy2:.4f}'
               f.write(str_w+"\n") # save
       
       f.write('\n')        
f.close()

file_name = 'errRateBest'


with open(file_name, 'w', encoding="utf-8") as f:
 for i in range(len(period)):
       f.write('%3s  %3d\n'%('period = ', period[i]))
       f.write('%3s, %2s, %2s, %2s, %2s, %3s, %2s, %2s, %2s, %2s\n'%('sig','nu','errTest','discrep','tie', 'sig','nu','errTest','discrep', 'tie')) 
       for j in range(len(init_sig)):   
             tmp = np.amin(all_err2[i,j,])
             tmpc = np.amin(all_err2c[i,j,])
             tmp_idx = np.where(all_err2[i,j,]==tmp)
             tmpc_idx = np.where(all_err2c[i,j,]==tmpc)
             idx = list(tmp_idx)[0][0]
             idxc = list(tmpc_idx)[0][0]
                
             nu_value = nu[idx]
             nuc_value = nu[idxc]
             str_w = f'{init_sig[j]:3d} & {nuc_value:3d} & {all_err2c[i,j,idxc]:.4f} & {all_discrepancy2c[i,j,idxc]:.4f} & {int(all_num2c[i,j,idxc]):3d} & & & {init_sig[j]:3d} & {nu_value:3d} & {all_err2[i,j,idx]:.4f} & {all_discrepancy2[i,j,idx]:.4f} & {int(all_num2[i,j,idx]):3d}\\\\'
             f.write(str_w+"\n") # save

 f.write("\n") 
f.close()