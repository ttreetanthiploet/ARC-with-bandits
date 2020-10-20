# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:25:23 2020

@author: ASUS
"""

import pickle
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt 

ncpu = 3
#the number of CPU used to run the simulation
Rep = 20
#the number of Simulation
Horizon = 100
#the number of trial in the simulation. *exclude initial observation
Bandit_type = "additional_info"
#the type of bandit. It could be either "additional_info" or "linear"
No_arm = 10
#the number of bandit.
Initial_n = np.array([5]*No_arm)
#the vector of the number of initial observation on each bandit before running algorithm.
Dimension_of_parameter = 10
if Bandit_type == "additional_info":
    Dimension_of_parameter = No_arm
#Dimension of parameter. This is required to identify for linear bandit. 
#For the bandit with additional information, this parameter will not be used.
prior_mean = np.zeros(Dimension_of_parameter)
#A vertor indicating the prior mean
prior_precision = np.identity(Dimension_of_parameter)
if Bandit_type == "additional_info":
    prior_precision = np.diag(prior_precision)
#Prior of precision. 
#This is a vector of prior inverse for the "additional information" bandit with  and it is a matrix for "linear" bandit.
#The prior precision for "linear" bandit is required to be invertible but  it is allowed to be zero for the "additional information" bandit
P = np.identity(No_arm)*1
P[0,:] = np.ones(No_arm)*5
#Set a parameter P. 
#In the  "additional information" bandit, P represents the precision matrix with dimension equal to the number of arm.
#We do not need to set P for a linear bandit
Var = np.ones(No_arm)
if Bandit_type == "additional_info":
    Var = np.diag(P)
#In the "linear bandit", vVar is a vector represents the inverse variance of each bandit.
B = np.identity(No_arm)
for i in range(No_arm):
    j = (i+1)%No_arm
    B[i,j] = 1
if Bandit_type == "additional_info":
    B = np.identity(No_arm)
#Set a feature vector in a form of matrix
reward_prem = np.zeros(No_arm)
reward_prem[0] = 1
#Set a reward premium for the bandit

################################################################################################################
#This is a set up for the bandit algorithm.
MAB_Alg = MAB_algorithm(Bandit_type, No_arm)
if Bandit_type == "additional_info":
    MAB_Alg.set_P_info(P)
if Bandit_type == "linear":
    MAB_Alg.set_feature_and_var_and_prior_var(Var, B)
    MAB_Alg.set_decision_prior_var(prior_precision)
MAB_Alg.set_reward_premium(reward_prem)
#This is a set up for the bandit environment.
MAB_Sample = MAB_environment(Bandit_type, No_arm)
if Bandit_type == "additional_info":
    MAB_Sample.set_P_info(P)
if Bandit_type == "linear":
    MAB_Sample.set_feature_and_var_and_prior_var(Var, B)
    
Theta = np.random.normal(0,1, size = (Dimension_of_parameter, Rep))
pickle.dump(Theta, open("Theta", "wb" ) )  
#Generate the parameter of the simulation
###############################################################################################################

#This is a command to run algorithm
def run(j):
    theta_in_sim = Theta[:,j]
    MAB_Sample.set_theta(theta_in_sim)
    #Set a parameter for the bandit    
    m,n = MAB_Sample.Initialise_information(Initial_n, prior_mean, prior_precision)
    #Generate initial information
    Decision_record = [None]*Horizon
    m_record = [None]*Horizon
    n_record = [None]*Horizon
    for t in range(Horizon):
        Decision_record[t] = Algorithm(m,n, t+1, Horizon)
        Observation = MAB_Sample.Sample(Decision_record[t])
        m, n = MAB_Alg.Parameter_update(m, n, Observation, Decision_record[t])
        (m_record[t], n_record[t]) = m,n      
    m_record = np.array(m_record)
    n_record = np.array(n_record)
    return (m_record, n_record, Decision_record)
###########################################################################################################
#List algorithms that we want to compare here.
Algo_list = []
Algo_save_list = []

Algo_list.append(MAB_Alg.ARC(0.05, 1-1/Horizon))
Algo_save_list.append("ARC_0_05")

Algo_list.append(MAB_Alg.KG(1000, 1-1/Horizon))
Algo_save_list.append("KG_1000")
#Choose an algorithm to consider and name the algorithm to save our data
###########################################################################################
for i in range(len(Algo_list)):
    Algorithm = Algo_list[i]
    H = Parallel(n_jobs=ncpu, max_nbytes='10M')(delayed(run)(j) for j in range(Rep))
    pickle.dump(H, open(Algo_save_list[i], "wb" ) )
    #We may also save_data in case we want to reuse our data
#########################################################################################
#Here is an example how to run a regret plot.
No_Alg = len(Algo_list)    
Data = [None]*No_Alg
for i in range(No_Alg):
    Data[i] = pickle.load( open(Algo_save_list[i], "rb" ) )
Theta_sim = pickle.load( open("Theta", "rb" ) )
#Download our saved data
Decision_record = np.zeros((len(Algo_list), Rep, Horizon))
for i in range(len(Algo_list)):
    for j in range(Rep):
        Decision_record[i,j,:] = Data[i][j][2]
Regret_record = np.zeros((len(Algo_list), Rep, Horizon))
#Arrage the recorded data

for j in range(Rep):    
    Decision_record_vector = Decision_record[:,j,:].reshape(No_Alg*Horizon)
    Decision_record_vector = Decision_record_vector.astype(int)
    Expected_reward = MAB_Alg.Compute_Expected_reward(np.array(np.matmul(B, Theta[:,j])), 1/Var)    
    Chosen_reward = Expected_reward[Decision_record_vector]
    Regret_record[:,j,:]  = max(Expected_reward) - Chosen_reward.reshape((No_Alg,Horizon))
#Compute expected expected regret
    
Cum_regret = np.cumsum(Regret_record, axis = 2)

q80_cumsum_regret = np.quantile(Cum_regret, 0.80, axis = 1)  
#compute an expected quantile for the regret
for i in range(No_Alg):
    plt.plot(np.arange(1,Horizon+1), q80_cumsum_regret[i,:], label = Algo_save_list[i])
plt.legend()
plt.ylabel('Expected-Expected Regret', fontsize = 12)
plt.xlabel('Trials', fontsize = 12)
plt.title('0.80 quantile of Expected-expected regret', fontsize = 12)

