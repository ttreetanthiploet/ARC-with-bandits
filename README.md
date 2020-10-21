# ARC-with-bandits
These are examples of Python script for the paper [Asymptotic Randomised Control with applications to bandits](https://arxiv.org/abs/2010.07252).

## How to run a simulation?

**1.** Run scripts: MAB_algorithm.py and MAB_environment.py to define a class of bandit algorithms and a class of bandit environment to generate samples.

**2.** Identify bandit type and the number of bandits. The type of bandit could either be "linear" or "additional_info". For example,
>```
>MAB_Alg = MAB_algorithm("linear", 3)
>MAB_Sample = MAB_environment("linear", 3)
>```
or
>```
>MAB_Alg = MAB_algorithm("additional_info", 3)
>MAB_Sample = MAB_environment("additional_info", 3)
>```


