# implement a variant of JMLR Alg.1 for smallteam data 
# here team score is defined as the sum oraverage of individual member scores
# the update rule contains both the sum-up and the average approaches

# Usage:  python3 halo2.py data.txt value 
# e.g. 
# python3 halo2.py sample.dat a
# python3 halo2.py sample.dat s

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import copy
import os
from sys import argv
from locale import atof, atoi

##################
# input argument #
##################
"""# input arguments
par0: data
par: a or s, with a corresponding to averageScore; and s corresponding to sumScore
"""

par0 = argv[1]
par = argv[2]

dataS = par0.split('.')
s0 = dataS[0]
print(s0)

#################
# change folder #
#################
folder =  'halo_' + s0 + '_' + par + 'Score'
folderS = folder+'/'

print(folder)
print(folderS)

try:
  os.mkdir(folderS)
except:
  pass

os.chdir(folder)

################
# data prepare #
################
data_name = '../' + par0

##################
### parameters ###
##################
init_mu = 25
init_sigma = init_mu/3
beta = init_sigma*0.5

##################
# some functions #
##################
def sigmoid(mu_w, mu_l,c_wl):
  x = (mu_w-mu_l)/c_wl
  return 1/(1+np.exp(-x))

#################
# main function #
#################
def calculate_ratings(data_name, start_mean, var, ave = 1, file_name=None, mode = 'Train', player_mean=None, player_var = None):
    if file_name==None: file_name = mode + '.txt'
    if mode =='Train':
        player_mean = defaultdict(lambda :(np.array([[start_mean]],dtype = 'float64')))
        player_var = defaultdict(lambda :(np.array([[var]],dtype = 'float64')))
        count,total,total_discrepancy = 0,0,0
        pred_tie = countx = county = counte = count0 = count1 = count2 = count3 = count4 = count5 = count6 = 0
    f2=open(data_name,'r')
    ngame = int(f2.readline())
    nplayer = int(f2.readline())
    for game in range(0, ngame):
        tmp = f2.readline()
        tmp = tmp.strip('\n')
        tmp = tmp.split(' ')
        n0 = int(tmp[1])
        n1 = int(tmp[2])
        team_num_players = [n0, n1]
        s0 = int(tmp[3])
        s1 = int(tmp[4])
        team_mu = [0, 0]
        team_var= [0, 0]
        n = n0 + n1
        player0i_list = []
        player1i_list = []
        player0i_score_list = []
        player1i_score_list = []
        #game information
        for player_i in range(0, n0):
            tmp2 = f2.readline()
            tmp2 = tmp2.strip('\n')
            tmp2 = tmp2.split(' ')
            player0i_list.append(tmp2[0])
            player0i_score_list.append(int(tmp2[1]))
            team_mu[0] = team_mu[0] + player_mean[tmp2[0]].item() #mu0
            team_var[0] = team_var[0] + player_var[tmp2[0]].item() #sigma0
        for player_i in range(0, n1):
            tmp2 = f2.readline()
            tmp2 = tmp2.strip('\n')
            tmp2 = tmp2.split(' ')
            player1i_list.append(tmp2[0])
            player1i_score_list.append(int(tmp2[1]))
            team_mu[1] = team_mu[1] + player_mean[tmp2[0]].item() #mu1
            team_var[1] = team_var[1] + player_var[tmp2[0]].item() #sigma0
        if min(team_num_players) == 0:
            continue

        if ave == 1:
            team_mu[0] = team_mu[0]/n0
            team_mu[1] = team_mu[1]/n1 
            team_var[0] = team_var[0]/n0/n0
            team_var[1] = team_var[1]/n1/n1   
        
        if par == 'a':
            s0 = s0/n0
            s1 = s1/n1
        if s0 > s1:
            winner_mean_old = team_mu[0]
            loser_mean_old = team_mu[1]
            r = [0, 1]
            p_flag = 0

        elif s0 < s1:
            winner_mean_old = team_mu[1]
            loser_mean_old = team_mu[0]
            r = [1, 0]
            p_flag = 1
        else:
            r = [0, 0]
        t_score = [s0, s1]
        team_player_list = [player0i_list, player1i_list]
        player_score_list = [player0i_score_list, player1i_score_list]

        #Predict_record is to be used in calibration plot 
        predict_record = 0
        for i in range(0,2):
            for j in range(i+1,2):
                if r[i] != r[j]:
                    total +=1 #count matches which are not ties
                    if (s0 - s1)*(team_mu[i] - team_mu[j]) > 0: 
                        count+=1 #count correct predictions
                        predict_record = 1
                        if (s0 - s1)*(n0 - n1) > 0: #big team wins
                            countx+=1
                        if (s0 - s1)*(n0 - n1) < 0: #small team wins
                            county+=1
                        elif(n0 - n1) == 0: #equal-sized match
                            counte += 1 
                    if (team_mu[i] - team_mu[j]) == 0: #result is tie game
                        pred_tie+=1
                    if (s0 - s1)*(team_mu[i] - team_mu[j]) < 0: #predict wrong
                        count0+=1
                    if (s0 - s1)*(n0 - n1) > 0: #among games with unequal team sizes, number of games where larger teams win
                        count1+=1
                    if (s0 - s1)*(n0 - n1) < 0: #big team lose
                        count2+=1
                    if (n0 - n1) !=0:
                        if (s0 - s1) != 0:
                            count3+=1 #number of games with unequal team size

                    if (n0 - n1) == 0:
                        if (s0 - s1) ==0:
                            count4+=1
                        elif (s0 - s1) != 0:
                            count5+=1
                    if ((s0 - s1) !=0):
                        count6+=1

        #player ratings update
        p_list = [0, 0]
        for i in range(0,2):
            Omega = 0
            Delta = 0
            for q in range(0,2):
                if i==q:
                    continue
                ciq = np.sqrt(team_var[i] + team_var[q] + 2*beta*beta)
                piq = sigmoid(team_mu[i], team_mu[q], ciq)
                p_list[i] = piq
                sigsq_to_ciq = team_var[i]/ciq
                s = 0
                if r[q] > r[i]:
                    s=1
                elif (r[i]==r[q]):
                    s=0.5
                Omega += sigsq_to_ciq*(s-piq)
                #gamma = np.sqrt(team_var[i])/ciq  # this gives acc=0.6603
                gamma = 1/2                     # this gives acc=0.6616
                Delta += gamma*sigsq_to_ciq/ciq*piq*(1-piq)
            for j in range(0, team_num_players[i]):
                sigmaijsq = player_var[team_player_list[i][j]].item()
                player_mean_old = player_mean[team_player_list[i][j]].item()
                if ave == 1:
                    alpha_iq = 1/team_num_players[i]
                if ave == 0:
                    alpha_iq = 1      
                player_mean_new = player_mean[team_player_list[i][j]].item() + alpha_iq*sigmaijsq/team_var[i]*Omega
                player_var[team_player_list[i][j]][0][0] = player_var[team_player_list[i][j]].item() * (1-alpha_iq*alpha_iq*sigmaijsq/team_var[i]*Delta)   
                player_mean[team_player_list[i][j]][0][0] = player_mean_new
            if r[0] != r[1]:
                discrepancy = p_list[p_flag]
                total_discrepancy += discrepancy
    f2.close()
    error = total-count
    acc = count/total
    err = count0/total
    c1toc12 = count1/(count1+count2)
    return player_mean, player_var, acc, err, (total,count,count0,pred_tie), (total_discrepancy/total), (count1, count2, count3, count4, count5, count6, countx, county, counte), c1toc12, total, count, acc, count3, count1, countx, county, counte

    # description of all count variables
    # total: number of games after removing tie games
    # all counts here exclude tie games
    # total = count + count0 + pred_tie, here "pred_tie" is the number of predicted tie
    # count: correct prediction
    # pred_tie: number of predicted ties; in JMLR, it is counted as wrong prediction
    # count1: among games with unequal team sizes, number of games where larger teams win
    # count2: among games with unequal team sizes, number of games where smaller teams win
    # count3: number of games with unequal team sizes
    # so, count3 = count1 + count2
    # count4: must be zero b/c all counts here exclude tie games
    # count5: number of games with equal team sizes
    # so, count3 + count5 = count6 = total
    # countx: among count (correct prediction), how many satisfy (s0-s1)(n0-n1)>0?
    #         that is, how many of them are "larger team win"
    # counte: among count (correct prediction), how many are of equal size 


############
# Run code #
############
file_name = 'accRate_ave.txt'
print(file_name)
est_mean, est_var, acc, err, num, discrep, c1toc6, c1toc12, o1, o2, o3, o4, o5, o6, o7, o8 = calculate_ratings(data_name, init_mu, init_sigma*init_sigma, mode = 'Train')


with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s,  %2s, %2s\n'%('accuracy','error',"(n,a,w,t)",'discrepancy'))
  str_w = '%.4f, %.4f, %s, %.4f, %s, %.4f' %(acc, err, num, discrep, c1toc6, c1toc12)
  f.write(str_w+"\n") # save
  f.write('%2s %2s %2s %2s %2s %2s %2s\n'%('total &', 'correct &', 'acc &', 'uneven-size &', 'even-size &', 'larger win &', 'smaller win'))
  str_w = '%d & %d & %.4f & %d & %d & %d & %d' %(o1, o2, o3, o4, o1-o4, o5, o4-o5)
  f.write(str_w+"\n") # save
  f.write("\n") # save
  f.write('%2s %2s %2s %2s\n'%('total correct&', 'larger-win correct&', 'smaller-win correct&', 'evensize correct'))
  str_w = '%d & %d & %d & %d' %(o2, o6, o7, o8)
  f.write(str_w+"\n") # save
  f.write('%2s %2s\n' %('acc-large win&','acc-smaller win'))
  str_w = '(%.4f) & (%.4f)' %(o6/o5, o7/(o4-o5))
  f.write(str_w + "\n")
  f.write('%2s %2s %2s \n'%('games&', 'uneven&', 'largerwin&'))
  str_w = '%d & %d & %d &' %(o1, o4, o5)
  f.write(str_w + "\n")
  f.write("\n")
  f.write('%2s %2s %2s \n'%('total&', 'n_W \\leq n_L&', 'n_W > n_L&'))
  str_w = '%d (%.4f) & %d & %d &' %(o2, o3, o2-o6, o6)
  f.write(str_w + "\n")
f.close()

#print results on the screen
acc_large = (o6 / o5) if o5 else 0.0
acc_small = (o7 / (o4 - o5)) if (o4 - o5) else 0.0
print(f"{'Total':>8} {'Correct':>9} {'Acc':>8} {'Larger Win':>12} {'Smaller Win':>13}")
print(f"{o1:>8d} {o2:>9d} {o3:>8.4f} {o5:>12d} {(o4 - o5):>13d}")

print()


print(f"{'Larger Win Corr':>17} {'Smaller Win Corr':>18}")
print(f"{o6:>17d} {o7:>18d}")

print()

print(f"{'Acc Large Win':>15} {'Acc Small Win':>15}")
print(f"{acc_large:>15.4f} {acc_small:>15.4f}")


file_name = 'accRate_sum.txt'
print(file_name)
est_mean, est_var, acc, err, num, discrep, c1toc6, c1toc12, o1, o2, o3, o4, o5, o6, o7, o8 = calculate_ratings(data_name, init_mu, init_sigma*init_sigma, ave=0, mode = 'Train')


with open(file_name, 'w', encoding="utf-8") as f:
  f.write('%3s, %2s,  %2s, %2s\n'%('sig','accTrain',"(n,a,t)",'discrepancy'))
  str_w = '%.4f, %.4f, %s, %.4f, %s, %.4f' %(acc, err, num, discrep, c1toc6, c1toc12)
  f.write(str_w+"\n") # save
  f.write('%2s %2s %2s %2s %2s %2s %2s\n'%('total &', 'correct &', 'acc &', 'uneven-size &', 'even-size &', 'larger win &', 'smaller win'))
  str_w = '%d & %d & %.4f & %d & %d & %d & %d' %(o1, o2, o3, o4, o1-o4, o5, o4-o5)
  f.write(str_w+"\n") # save
  f.write("\n") # save
  f.write('%2s %2s %2s %2s\n'%('total correct&', 'larger-win correct&', 'smaller-win correct&', 'evensize correct'))
  str_w = '%d & %d & %d & %d' %(o2, o6, o7, o8)
  f.write(str_w+"\n") # save
  f.write('%2s %2s\n' %('acc-large win&','acc-smaller win'))
  str_w = '(%.4f) & (%.4f)' %(o6/o5, o7/(o4-o5))
  f.write(str_w + "\n")
  f.write('%2s %2s %2s \n'%('games&', 'uneven&', 'largerwin&'))
  str_w = '%d & %d & %d &' %(o1, o4, o5)
  f.write(str_w + "\n")
  f.write("\n")
  f.write('%2s %2s %2s \n'%('total&', 'n_W \\leq n_L&', 'n_W > n_L&'))
  str_w = '%d (%.4f) & %d & %d &' %(o2, o3, o2-o6, o6)
  f.write(str_w + "\n")


#print results on the screen
acc_large = (o6 / o5) if o5 else 0.0
acc_small = (o7 / (o4 - o5)) if (o4 - o5) else 0.0
print(f"{'Total':>8} {'Correct':>9} {'Acc':>8} {'Larger Win':>12} {'Smaller Win':>13}")
print(f"{o1:>8d} {o2:>9d} {o3:>8.4f} {o5:>12d} {(o4 - o5):>13d}")

print()


print(f"{'Larger Win Corr':>17} {'Smaller Win Corr':>18}")
print(f"{o6:>17d} {o7:>18d}")

print()

print(f"{'Acc Large Win':>15} {'Acc Small Win':>15}")
print(f"{acc_large:>15.4f} {acc_small:>15.4f}")
f.close()