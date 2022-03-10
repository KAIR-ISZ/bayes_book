#!/usr/bin/env python
# coding: utf-8

# # Data analytics
# 
# ## Exploring distributions
# 
# ### dr hab. inż. Jerzy Baranowski, Prof. AGH

# In[1]:


import scipy.stats as stats
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

light="#FFFCDC"
light_highlight="#FEF590"
mid="#FDED2A"
mid_highlight="#f0dc05"
dark="#EECA02"
dark_highlight="#BB9700"
green="#00FF00"
light_grey="#DDDDDD"


# ## Managing expectations
# 
# - Main point of probabilistic compoutation is to compute an expectation of certain function of parameters. 
# 
# - It generally covers all kind of statistics
# 
# - Has many beneficial properties

# ## Expectations
# 
# In general case, function of parameter $q\in Q$: $f(q)$  with respect to a probability distribution (mass function) $\pi(q)$ has expectation given by
# 
# $$
# \mathbb{E}_{\pi}[f] = \int_{Q} \mathrm{d} q \, \pi(q) \, f(q).
# $$
# 
# or in discrete case
# 
# $$
# \mathbb{E}_{\pi}[f] = \sum_{q \in Q} \pi(q) \, f(q).
# $$

# ## How to compute expectations?
# 
# Analytic integration is practically impossible. 
# 
# We are left with quadratures, for ex. Euler
# $$
# \mathbb{E}_{\pi}[f] \approx 
# \sum_{n = 1}^{N} (\Delta q)_{n} \, \pi(q_{n}) \, f(q_{n}).
# $$
# 

# ## How to compute expectations?
# 
# Other option is exact sampling, leading to so called Monte Carlo estimators.
# 
# If we can generate set of samples $\{ q_{1}, \ldots, q_{N} \} \in Q$, such that
# 
# $$\hat{f}_{N}^{\text{MC}} = \frac{1}{N} \sum_{n = 1}^{N} f(q_{n}),$$
# 
# asymptotically converges
# 
# $$
# \lim_{N \rightarrow \infty} \hat{f}_{N}^{\text{MC}} = \mathbb{E}_{\pi}[f].
# $$
# 
# Then we have an exact sampling procedure
# 

# ## Monte Carlo estimators
# 
# Provided, that samples are generated properly we can quantify estimator error
# 
# $$
# \frac{ \hat{f}_{N}^{\text{MC}} - \mathbb{E}_{\pi}[f] }
# {\text{MC-SE}_{N}[f] } 
# \sim \mathcal{N}(0, 1),
# $$
# 
# With Monte Carlo Standard Error given by
# $$
# \text{MC-SE}_{N}[f] 
# = \sqrt{ \frac{ \text{Var}_{\pi}[f]}{N} }.
# $$

# ## Motivational example
# - We return to Mass Effect experiment using grid approximation
# 

# <img src="img/masseffect.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; EA &amp; Bioware </div>
# 

# ## Grid approximation
# 1. Define the grid. This means you decide how many points to use in estimating the posterior, and then you make a list of the parameter values on the grid.
# 2. Compute the value of the prior at each parameter value on the grid.
# 3. Compute the likelihood at each parameter value.
# 4. Compute the unstandardized posterior at each parameter value, by multiplying the prior by the likelihood.
# 5. Finally, standardize the posterior, by dividing each value by the sum of all values.

# In[2]:


def posterior_grid_approx(grid_points=5, success=6, tosses=9):
    
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform
   
    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior


# In[3]:


posterior_grid_approx()


# ## Grid approximation example

# In[4]:


f3, (ax1, ax2) = plt.subplots(1, 2, sharex=True,figsize=(7,4),tight_layout=True)
grid5=posterior_grid_approx()
grid20=posterior_grid_approx(20)
ax1.plot(grid5[0],grid5[1],marker='o',color=dark)
ax1.set_title('5 points')
ax1.set_xlabel('probability of water')
ax1.set_ylabel('posterior probability')
ax1.set_yticks([])
ax2.plot(grid20[0],grid20[1],marker='o',color=dark)
ax2.set_title('20 points')
ax2.set_xlabel('probability of water')
ax2.set_yticks([])
plt.show()


# In[5]:


f3


# ## Summarizing by sampling
# - The easiest way to get information about even complicated posteriors is to simulate data that correspond to it and get parameter estimates from sampling.
# - For single parameter problems the easiest way is to use inverse cumulative distribution function 
# - Grid approximation is more universal

# In[6]:


def posterior_grid_approx(grid_points=5, success=7, tosses=11,prior_sel=1):
    
    # define grid
    p_grid = np.linspace(0, 1, grid_points)

    # define prior
    if prior_sel==1:
        prior = np.repeat(5, grid_points)  # uniform
    elif prior_sel==2:
        prior = (p_grid >= 0.5).astype(int)  # truncated
    elif prior_sel==3:    
        prior = np.exp(- 5 * abs(p_grid - 0.5))  # double exp 
    else:
        raise ValueError('Unsuported prior selection')
    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(success, tosses, p_grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return p_grid, posterior


# In[7]:



p_grid, posterior=posterior_grid_approx(100,6,9,prior_sel=3)
p_grid, prior=posterior_grid_approx(100,0,0,prior_sel=3)
fig,ax = plt.subplots(1, 1, figsize=(7,4),tight_layout=True)
ax.plot(p_grid, posterior,color=dark,label='Posterior')
ax.plot(p_grid, prior,color=mid,label='Prior')


ax.set_xlabel('probability of water')
ax.set_ylabel(' probability')
ax.set_yticks([])
ax.legend()
plt.show()


# ## Sampling from grid posterior

# In[8]:


np.random.seed(44)
samples=np.random.choice(p_grid,size=10000,replace=True,p=posterior)

fig,ax = plt.subplots(1, 1, figsize=(7,4),tight_layout=True)
ax.scatter(range(10000),samples,color=dark,edgecolor=dark_highlight)
ax.set_xlabel('sample number')
ax.set_ylabel(r'value of $\theta$')
plt.show()


# ## Weakly informative priors
# We characterize a prior distribution as weakly informative if it is proper but is set up so that the information it does provide is intentionally weaker than whatever actual prior knowledge is available. 
# - Make uninformative more complicated
# - Make informative less complicated

# In[9]:


fig,ax = plt.subplots(1, 1, figsize=(7,4),tight_layout=True)
ax.hist(samples,bins=25,density=True,color=dark,edgecolor=dark_highlight)
ax.set_xlabel(r'value of $\theta$')
plt.show()


# In[10]:


# posterior probability where p < 0.5
print(np.sum( posterior[ p_grid < 0.5 ] ))


# In[11]:


# same by sampling
print(np.sum( samples < 0.5 ) / 1e4)


# In[12]:


# intervals of interest
print(np.sum( (samples > 0.5) & (samples < 0.75) ) / 1e4)


# In[13]:


# quantiles
np.quantile(samples, [0.1,0.9])


# ## Prior predictive distribution

# In[14]:


pr_samples=np.random.choice(p_grid,size=10000,replace=True,p=prior)
pr_pr_d_samples=np.random.binomial(1,pr_samples)

print('Mean rate of success = {}'.format(np.sum(pr_pr_d_samples)/1e4))


# In[15]:


fig,ax = plt.subplots(1, 1, figsize=(7,4),tight_layout=True)
ax.hist(pr_pr_d_samples,bins=[-.1,.1,.9,1.1],density=True,color=dark,edgecolor=dark_highlight)
ax.set_xlabel('outcome')
ax.set_xticks([0,1])
ax.set_yticks([])
plt.show()


# ## Posterior predictive distribution

# In[16]:


post_samples=np.random.choice(p_grid,size=10000,replace=True,p=posterior)
post_pr_d_samples=np.random.binomial(1,post_samples)
print('Mean rate of success = {}'.format(np.sum(post_pr_d_samples)/1e4))


# In[17]:


fig,ax = plt.subplots(1, 1, figsize=(7,4),tight_layout=True)
ax.hist(post_pr_d_samples,bins=[-.1,.1,.9,1.1],density=True,color=dark,edgecolor=dark_highlight)
ax.set_xlabel('outcome')
ax.set_xticks([0,1])
ax.set_yticks([])
plt.show()


# In[18]:


post_pr_d_samples


# ## Do grid approximations generalize?
# It depends

# <img src="img/grid_density.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Sometimes geometry is difficult

# <img src="img/grid_negligible.png" alt="drawing" width="500"/>
# 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## What contributes to expectation?
# Expectation is an integral
# 
# $$
# \mathbb{E}_{\pi}[f] = \int_{Q} \mathrm{d} q \, \pi(q) \, f(q).
# $$
# 
# Intuitively, wherever distribution $\pi(q)$ is large, it should contribute the most, in particular next to maximum (mode).

# <img src="img/conc_of_meas_anal_1.png" alt="drawing" width="400"/>
# 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## But what about the volume?
# 
# $dq$ is also under the integral, and volume rises with dimension
# <img src="img/box-1d.png" alt="drawing" width="150"/>
# <img src="img/box-2d.png" alt="drawing" width="500"/>
# <img src="img/box-3d.png" alt="drawing" width="700"/>
# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>

# ## Volume rises exponentially with dimension

# <img src="img/conc_of_meas_anal_2.png" alt="drawing" width="400"/>
# 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## It is the product that counts
# 

# <img src="img/conc_of_meas_anal_3.png" alt="drawing" width="400"/>
# 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Typical set
# What contributes the most to the expectation are the values from the typical set

# <img src="img/conc_of_meas_anal_4.png" alt="drawing" width="400"/>
# 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Concentration of measure
# 
# Typical set, is a "fuzzy surface" that is located progressively away from the mode with the rise of dimension.

# <img src="img/typical_set.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Typical set is where we should sample from

# <img src="img/typical_set_samples.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Computational algorithms for probabilistic computing
# 
# - Point estimators
# - Laplace approximation
# - Variational approximation
# - Monte Carlo estimators
# - Markov Chain Monte Carlo 

# ## Modal estimators 
# This approach searches for the maximal value of probability distribution, in order to obtain approximation of expected value

# <img src="img/good_mode.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Issues
# - skewed distributions have maxima far from expectations
# - problems with uncertainty quantisation

# <img src="img/bad_mode.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Laplace estimator
# 
# Main idea is to find the maximal value, and fit a Gaussian distribution with a mean in it, and covariance obtained by second order Taylor approximation.
# 
# Expectation values can then estimated with Gaussian integrals,
# $$
# \mathbb{E}_{\pi} \! \left[ f \right]
# \approx 
# \int_{Q} \mathrm{d} q \, \mathcal{N} \! \left( q \mid \mu, \Sigma \right) \,
# f \! \left( q \right),
# $$

# ## If distribution is relatively close to Gaussian, typical set is well approximated

# <img src="img/laplace.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Variational approximation
# 
# - The main idea is to approximate the posterior with functions, that can be easily sampled from (or their combination). 
# 
# - Such approximation is realized by minimization of a function called divergence, which measures how differetnt candidate and probability distribution are from one another.
# 
# - In practice it is done by minimizing certain bound on the divergence.

# ## Multimodality of variational approximation
# 
# It can happen, that significantly different candidates have similar divergences, that causes optimization problem to be multimodal
# 
# 

# <img src="img/degenerate_fits.png" alt="drawing" width="1000"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Over and under fitting of the typical set
# 
# <img src="img/overestimated_var.png" alt="drawing" width="500"/>
# <img src="img/underestimated_var.png" alt="drawing" width="500"/>
# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>

# ## Monte Carlo sampling

# <img src="img/typical_set_samples.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Issues with Monte Carlo 
# 
# - It is easy to sample from known distributions, especially low dimensional or normal
# - Complicated distributions are an issue
# - Importance sampling is an option
#     - Sample from something that you know (proposal distribution)
#     - Correct with properly chosen weights
#     - Strongly depends on quality of proposal

# ## Markov Chain Monte Carlo
# 

# <img src="img/typical_set_markov_chain.png" alt="drawing" width="500"/> 

# <div style="text-align: right"> <span style="font-size:.3em;">Image &copy; <a href="https://betanalpha.github.io"> Michael Betancourt</a></span> </div>
# 

# ## Extra reading
# 
# [Probabilistic Computation by Michael Betancourt](https://betanalpha.github.io/assets/case_studies/probabilistic_computation.html#1_representation_with_computational_taxation)

# In[ ]:




