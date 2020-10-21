# ARC-with-bandits
These are examples of Python script for the paper [Asymptotic Randomised Control with applications to bandits](https://arxiv.org/abs/2010.07252).

## How to run a simulation?
**1.** Import relevant modules.
>```
>import numpy as np
>import scipy.optimize
>from scipy.stats import norm
>```

**2.** Run scripts: MAB_algorithm.py and MAB_environment.py to define a class of bandit algorithms and a class of bandit environment to generate samples.

**3.** Identify bandit type and the number of bandits. The type of bandit could either be "additional_info" or  "linear". For example,
>```
>MAB_Alg = MAB_algorithm("additional_info", 3)
>MAB_Sample = MAB_environment("additional_info", 3)
>```
or
>```
>MAB_Alg = MAB_algorithm("linear", 3)
>MAB_Sample = MAB_environment("linear", 3)
>```

**4.** Set a precision matrix (P) for "additional_info" bandit or set a feature, a variance inverse and prior precision (B, V, Sigma_0) for "linear" bandit. For example,
>```
>P = np.identity(3)
>MAB_Alg.set_P_info(P)
>MAB_Sample.set_P_info(P)
>```
or
>```
>B = np.array([[1,1,0], [1,0,1], [0,1,1]])
>V = np.ones(3)
>Sigma_0 = np.identity(3)
>MAB_Alg.set_feature_and_var_and_prior_var(V, B)
>MAB_Alg.set_decision_prior_var(Sigma_0)
>MAB_Sample.set_feature_and_var_and_prior_var(V, B)
>```

**5.** Set a reward premium for each bandit. For example,
>```
>MAB_Alg.set_reward_premium(np.array([0.5,0,0]))
>```

**6.** Set a parameter $\theta$ for a system of bandits. For example,
>```
>MAB_Sample.set_theta(np.array([3,2,1]))
>```

**6.** Set a prior decision and generate an initail information. For example,
>```
>prior_mean = np.array([0,0,0])
>prior_precision = np.identity(3)
>Initial_n = np.array([5,5,5])
>m,n = MAB_Sample.Initialise_information(Initial_n, prior_mean, prior_precision)
>#We shall think of d in the paper as 1/n 
>```




