# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:22:53 2020

@author: ASUS
"""

import numpy as np
import scipy.optimize
from scipy.stats import norm

class MAB_algorithm():
    def __init__(self, Bandit_type, arm_size):
        #Bandit_type could be either "linear" or "additional_info"
        if Bandit_type=="linear" or Bandit_type=="additional_info":
            self.bandit_type = Bandit_type
            self.reward_premium = np.zeros(No_arm)
            self.No_arm = arm_size
        else:
            print("The bandit type is not correct. Bandit type must either be linear or additional info.")
               
    def set_P_info(self, P):
        if self.bandit_type == "additional_info":
            self.Information_matrix = np.array(P)
            self.Diag_P = np.diag(self.Information_matrix)  
        else: 
            print("The bandit type is not correct. Bandit type must must be additional info.")
    
    def set_feature_and_var_and_prior_var(self, P, B):
        if self.bandit_type == "linear":
            self.P = np.array(P)
            self.B = np.matrix(B)
        else: 
            print("The bandit type is not correct. Bandit type must must be linear.")
        
    def set_decision_prior_var(self,Sigma_0_inv):
        if self.bandit_type == "linear":
            self.Sigma_0_inv = Sigma_0_inv
        else: 
            print("The bandit type is not correct. Bandit type must must be linear.")

            
    def set_reward_premium(self, r):
        self.reward_premium = r
    
    def Reward(self, choice, Observation):
        return norm.cdf(Observation) - self.reward_premium[choice]
    
    def nu(self, a, Lambda):
        a = np.array(a)
        #print(a)
        ref = max(a)
        W = np.exp((a-ref)/Lambda)
        K = sum(W)        
        return  W/K
    
    def Sigma(self, d):
        S = self.Sigma_0_inv
        for i in range(len(self.P)):
            b_i = np.matrix(self.B[i])
            S = S + self.P[i]/d[i]*np.matmul(np.transpose(b_i), b_i)            
        Sigma = np.linalg.inv(S)
        return Sigma
    
    def Sigma_inv(self, d):
        S = self.Sigma_0_inv
        for i in range(len(self.P)):
            b_i = np.matrix(self.B[i])
            S = S + self.P[i]/d[i]*np.matmul(np.transpose(b_i), b_i)
        return S
    
    def Update_Sigma_inv(self, Sigma_inv, i):
        b_i = np.matrix(self.B[i])
        Sigma_inv_update = Sigma_inv + self.P[i]*np.matmul(np.transpose(b_i), b_i)
        return Sigma_inv_update
    
    def Posterior_effect_outcome(self,d):
        S = np.array(np.matmul(np.matmul(self.B, self.Sigma(d)), np.transpose(self.B)))
        return S
    
    def Gaussian_density(self,z):
        return 1/np.sqrt(2*np.pi)*np.exp(-z**2/2)
    
    def L_input(self, m, d, Lambda):
        Sigma = self.Sigma(d)
        B = self.B
        P = self.P
        S = np.array(np.matmul(np.matmul(B, Sigma), np.transpose(B)))
        v_d = np.array(1 + np.diag(S) + 1/P)
        b_m = np.array(np.matmul(B,m))
        h = self.Gaussian_density(b_m/np.sqrt(v_d))/np.sqrt(v_d)
        g = h*b_m/v_d
        S_sq = S**2
        P_over_1d = P/(1+d) 
        diag_s_p = np.array(np.diag(S) + 1/P)
        coeff_first_term = P_over_1d - 1/diag_s_p
        return (coeff_first_term, S_sq, S, g, h, diag_s_p)
    
    def L_compute(self, a, Lambda, coeff_first_term, S_sq, S, g, h, diag_s_p):        
        nu_g = self.nu(a, Lambda)*g        
        First_term= 1/2*coeff_first_term * np.matmul(nu_g,  S_sq)     
        nu_h = self.nu(a, Lambda)*h
        nu_h2 = self.nu(a, Lambda)*h**2
        second_term = 1/(2*Lambda)*(1/diag_s_p)*(np.matmul(nu_h2, S_sq) - np.matmul(nu_h, S)**2)
        L = First_term + second_term
        return L[0]
    
    def L(self, a, m, d, Lambda):
        if self.bandit_type == "additional_info":
            Term_in_pdf = m/np.sqrt(1 + d + 1/self.Diag_P)
            Term_outside_pdf = self.nu(a, Lambda)*(1-self.nu(a, Lambda))/(1 + d + 1/self.Diag_P)
            Future_term =  Term_outside_pdf*(self.Gaussian_density(Term_in_pdf))**2
            D_copy = np.array([d]*len(d))
            minus_B = (D_copy**2)*self.Information_matrix*(1+D_copy*self.Information_matrix)
            L = (1/(2*Lambda))*np.matmul(minus_B, Future_term)     
        elif self.bandit_type == "linear":
            (coeff_first_term, S_sq, S, g, h, diag_s_p) = self.L_input(m, d, Lambda)
            L = self.L_compute(a, Lambda, coeff_first_term, S_sq, S, g, h, diag_s_p)
        return L
    
    def f(self,m, d):
        if self.bandit_type == "additional_info":
            Term_in_pdf = m/np.sqrt(1 + d + 1/self.Diag_P)
            f = norm.cdf(Term_in_pdf) - self.reward_premium
        elif self.bandit_type == "linear":
            Sigma = self.Sigma(d)
            B = self.B
            P = self.P
            S = np.array(np.matmul(np.matmul(B, Sigma), np.transpose(B)))
            v_d = np.array(1 + np.diag(S) + 1/P)
            b_m = np.array(np.matmul(B,np.transpose(m)))
            f = norm.cdf(b_m/np.sqrt(v_d))
            f = f[0]
        return f

    def f_max(self,m, d):
        Sigma = self.Sigma(d)
        B = self.B
        P = self.P
        S = np.array(np.matmul(np.matmul(B, Sigma), np.transpose(B)))
        v_d = np.array(1 + np.diag(S) + 1/P)
        b_m = np.array(np.matmul(B,np.transpose(m)))
        v_d = np.transpose(np.array([v_d]*np.shape(m)[0]))
        f = norm.cdf(b_m/v_d**(1/2))
        maxf = f.max(axis = 0)
        return maxf     

    def alpha(self, m,d, Lambda, beta):
        f = self.f(m, d)       

        def root_search(a):
            return f + (beta/(1-beta))*self.L(a,m,d, Lambda) - a
        
        a_root = scipy.optimize.root(root_search, f + (beta/(1-beta))*self.L(f,m,d, Lambda)).x
        return a_root
    
    def Compute_Expected_reward(self, theta, variance_vector):
        #Here we will compute E(erf(Y)) where Y is N(theta_i, variance_vector_i)
        Position = theta/np.sqrt(1 + variance_vector)
        return norm.cdf(Position) - self.reward_premium
    
    def Thompson_simulation(self):
        if self.bandit_type == "additional_info":
            def Algorithm(m,n, t, T):
                Posterior_sample = np.random.normal(m, np.sqrt(1/n))
                Posterior_reward = self.Compute_Expected_reward(Posterior_sample, 1/self.Diag_P) 
                random_choice = np.argmax(Posterior_reward)
                return random_choice
        elif self.bandit_type == "linear":
            def Algorithm(m,n, t, T):
                Posterior_var = self.Sigma(1/n)
                Posterior_sample = np.random.multivariate_normal(m, Posterior_var)
                Posterior_outcome_parameter = np.matmul(self.B, Posterior_sample)  
                Posterior_reward = self.Compute_Expected_reward(Posterior_outcome_parameter, 1/self.P) 
                random_choice = np.argmax(Posterior_reward)
                return random_choice        
        return Algorithm            
    
    def Greedy(self, epsilon):
        def Algorithm(m,n, t, T):
            u = np.random.uniform(0,1)
            if u < epsilon:
                random_choice = np.random.choice(range(len(m)))
            else:
                Expected_reward = self.f(m, 1/n)
                random_choice = np.argmax(Expected_reward)
            return random_choice
        return Algorithm            
     
    def Bayes_UCB(self, c):
        if self.bandit_type == "additional_info":
            def Algorithm(m,n, t, T):
                d =   1/n
                Bayes_UCB_index = norm.cdf(m+(d+1/self.Diag_P)**(1/2)*norm.ppf(1-1/(t*(np.log(T))**c))) - self.reward_premium            
                random_choice = np.argmax(Bayes_UCB_index)
                return random_choice
        elif self.bandit_type == "linear":
            def Algorithm(m,n, t, T):
                mean = np.matmul(self.B, np.transpose(m))  
                mean = np.array(mean)[0]         
                variance = np.diag(self.Posterior_effect_outcome(1/n)) + 1/self.P    
                Bayes_UCB_index = norm.cdf(mean+(variance)**(1/2)*norm.ppf(1-1/(t*(np.log(T))**c)))
                random_choice = np.argmax(Bayes_UCB_index)
                return random_choice
        return Algorithm      
    
    def ARC(self, k, beta):
        def Algorithm(m,n, t, T):
            f = self.f(m,1/n)
            weight = self.nu(f,1)
            norm = weight/n**2
            Lambda = k*np.sqrt(sum(norm))
            Realised_reward = self.alpha(m,1/n, Lambda, beta)
            Prob_simplex = self.nu(Realised_reward, Lambda)
            random_choice = np.random.choice(range(len(m)), p=Prob_simplex)
            return random_choice
        return Algorithm

    def KG(self, No_Monte_Carlo, beta):
        if self.bandit_type == "additional_info":
            def Algorithm(m,n, t, T):
                Z = np.random.standard_normal(size = (No_Monte_Carlo, len(m)))
                d = 1/n
                Expected_reward = np.zeros(len(m))
                for i in range(len(m)):
                    D = 1/(n + self.Information_matrix[i,:])
                    sigma = np.sqrt(d-D)
                    M = m + Z*sigma
                    F = self.f(M,D)
                    max_F = F.max(axis = 1)
                    Expected_reward[i] = np.mean(max_F)
                KG_index = self.f(m,d) + beta/(1-beta)*Expected_reward
                random_choice = np.argmax(KG_index)          
                return random_choice
        elif self.bandit_type == "linear":
            def Algorithm(m,n, t, T):
                Z = np.random.standard_normal(size = (No_Monte_Carlo, len(m)))
                d = 1/n
                Sigma_inv = self.Sigma_inv(d)
                Outcome_var = np.diag(self.Posterior_effect_outcome(d))
                Expected_reward = np.zeros(len(n))
                for i in range(len(n)):
                    n_forward = n
                    n_forward[i] = n_forward[i]+1
                    sigma = self.P[i]*Outcome_var[i]**(1/2)*np.matmul(self.B[i], np.linalg.inv(self.Update_Sigma_inv(Sigma_inv, i)))
                    M = m +np.matmul(Z, np.transpose(sigma))       
                    max_F = self.f_max(M, 1/n_forward)
                    Expected_reward[i] = np.mean(max_F)
                    
                KG_index = self.f(m,d) + beta/(1-beta)*Expected_reward
                random_choice = np.argmax(KG_index)   
                return random_choice
        return Algorithm
            
    def IDS(self, No_Monte_Carlo):
        def Algorithm(m,n, t, T):
            if self.bandit_type == "additional_info":
                d = 1/n
                Z = np.random.standard_normal(size = (No_Monte_Carlo, len(m)))
                Theta_hat = m+Z*d
                Expected_reward = self.Compute_Expected_reward(Theta_hat, 1/self.Diag_P)
                Regret = np.array([Expected_reward.max(axis = 1)]*len(m)).transpose() - Expected_reward
                delta = np.mean(Regret, axis = 0)
                D_copy = np.array([d]*len(d))
                Entropy_change = np.log(1+D_copy*self.Information_matrix)
                g = np.sum(Entropy_change, axis = 1)            
            elif self.bandit_type == "linear":            
                Posterior_var = self.Sigma(1/n)
                Posterior_sample = np.random.multivariate_normal(m, Posterior_var, size = No_Monte_Carlo)
                Posterior_outcome_parameter = np.matmul(self.B, np.transpose(Posterior_sample))
                Copy_outcome_var =  np.transpose(np.array([1/self.P]*No_Monte_Carlo))
                Expected_reward = self.Compute_Expected_reward(Posterior_outcome_parameter, Copy_outcome_var) 
                Regret = np.array([Expected_reward.max(axis = 0)]*len(n)) - Expected_reward
                delta = np.mean(Regret, axis = 1)
                g = np.zeros(len(n))
                Sigma_inv = self.Sigma_inv(1/n)
                for i in range(len(n)):
                    New_sigma_inv = self.Update_Sigma_inv(Sigma_inv, i)
                    g[i] = 1/2*(-np.log(np.linalg.det(Sigma_inv)) + np.log(np.linalg.det(New_sigma_inv)))   
            
            Avalable_p = []
            record_index = []
            for i in range(len(m)):
                for j in range(i+1, len(m)):
                    if abs(g[i]-g[j]) > 10**(-10):
                        p = 2*g[j]/(g[i]-g[j]) - delta[j]/(delta[i]-delta[j])
                        if (p > 0) & (p < 1):
                            Avalable_p.append(p)
                            record_index.append([i,j])
            for i in range(len(m)):
                Avalable_p.append(1)
                record_index.append([i,i])
            
            random_prob = np.array(Avalable_p)
            Record_index = np.array(record_index)
            
            random_regret = random_prob*delta[Record_index[:,0]] + (1-random_prob)*delta[Record_index[:,1]]
            random_information = random_prob*g[Record_index[:,0]] + (1-random_prob)*g[Record_index[:,1]]
            inform_ratio = random_regret**2/random_information
            best_IDS = np.argmin(inform_ratio)
            Candidate_prob = random_prob[best_IDS]
            Candidate = Record_index[best_IDS, :]
            random_choice = np.random.choice(Candidate, p=[Candidate_prob, 1-Candidate_prob])       
            return random_choice
        return Algorithm
    
    def Parameter_update(self, m, n, Observation, choice):
        if self.bandit_type == "additional_info":
            Observation_variance = self.Information_matrix[choice,:]
            m_update = (n*m + Observation_variance*Observation)/(n+Observation_variance)
            n_update = n + Observation_variance
        elif self.bandit_type == "linear":
            Sigma_inv = self.Sigma_inv(1/n)
            First_term = np.matmul(m, Sigma_inv)            
            Second_term =  Observation*self.B[choice]*self.P[choice]
            Total_sum = First_term + Second_term
            Sigma_inv = self.Update_Sigma_inv(Sigma_inv, choice)
            m_update = np.matmul(Total_sum, np.linalg.inv(Sigma_inv))
            m_update = np.array(m_update)[0]
            n_update = n
            n_update[choice] = n[choice] + 1
        return (m_update, n_update)

        