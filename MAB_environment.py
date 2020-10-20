# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:04:06 2020

@author: ASUS
"""

import numpy as np

class MAB_environment():
    def __init__(self, Bandit_type, arm_size):
        #Bandit_type could be either "linear" or "additional_info"
        if Bandit_type=="linear" or Bandit_type=="additional_info":
            self.bandit_type = Bandit_type
            self.reward_premium = np.zeros(arm_size)     
            self.No_arm = arm_size
        else:
            print("The bandit type is not correct. Bandit type must either be linear or additional info.")
            
    def set_P_info(self, P):
        if self.bandit_type == "additional_info":
            self.Information_matrix = np.array(P)
            self.Diag_P = np.diag(self.Information_matrix)  
        else: 
            print("The bandit type is not correct. Bandit type must must be additional info.")
    
    def set_feature_and_var_and_prior_var(self, P, B, ):
        if self.bandit_type == "linear":
            self.P = np.array(P)
            self.B = np.matrix(B)
        else: 
            print("The bandit type is not correct. Bandit type must must be linear.")
            
    def set_theta(self, theta):
        self.theta = theta
        if self.bandit_type == "linear":            
            self.outcome_parameter = np.array(np.matmul(self.B, self.theta))[0]
            
    def Initialise_information(self, Initial_n, prior_mean, prior_precision):
        if self.bandit_type == "additional_info":
            Initial_n = np.array(Initial_n)
            m = np.random.normal(self.theta, np.sqrt(1/Initial_n))
            new_m = (Initial_n*m + prior_precision*prior_mean)/(Initial_n + prior_precision)
            new_n = Initial_n + prior_precision
            self.Initial_state = (new_m, new_n)
        elif self.bandit_type == "linear": 
            Initial_n = np.array(Initial_n)
            M = np.matrix(prior_mean)
            Sigma_inv = prior_precision
            for i in range(len(Initial_n)):
                First_term = np.matmul(M, Sigma_inv)                
                Total_outcome = Initial_n[i]*self.outcome_parameter[i] + np.sqrt(Initial_n[i])*np.random.standard_normal()/np.sqrt(self.P[i])
                Second_term = self.B[i]*self.P[i]*Total_outcome 
                Total_sum = First_term + Second_term
                b_i = np.matrix(self.B[i])
                Sigma_inv = Sigma_inv + self.P[i]*Initial_n[i]*np.matmul(np.transpose(b_i), b_i)
                M = np.matmul(Total_sum, np.linalg.inv(Sigma_inv))
            new_m = np.array(M)[0]
            new_n = Initial_n
        return(new_m, new_n)
            
    def Sample(self, choice):
        if self.bandit_type == "additional_info":
            Observation_variance = self.Information_matrix[choice,:]
            Outcome = np.random.normal(self.theta[Observation_variance != 0], np.sqrt(1/Observation_variance[Observation_variance != 0]))
            Observation = np.zeros(self.No_arm)
            Observation[Observation_variance != 0] = Outcome
        elif self.bandit_type == "linear":
            Observation = self.outcome_parameter[choice]+np.random.standard_normal()/np.sqrt(self.P[choice])
        return Observation
        