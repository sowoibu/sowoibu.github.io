---
layout: post
title: Trust region policy optimization
use_math: true
summary: TRPO aims steady improvement of policy.
---
<head>
</head>

Summary of Trust Region Policy Optimization.

<h1>
Vanilla Policy Gradient   
</h1>
<br>
From policy gradient theorem, policy gradient is defined as 

$$\begin{array}{l}
\nabla_\theta J(\theta)=\mathbb{E}_{s'\sim d_\mu^{\pi_\theta}}\mathbb{E}_{a\sim \pi_\theta(a|s')}\nabla_\theta \log {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)
\end{array}$$
<br>
Policy gradient provides local estimator of	the gradient of the expected reward. However, large step of updates using policy gradient often results in policy collapse.
TRPO is a method that maintains stability while optimizing policies.
<br><br><br>


<h1>
Trust Region Policy Optimization
</h1>
<br>
In TRPO, what we want is a continuous improvement of policy. The monotonic improvement of policy is guaranteed by setting Trust Region based on current policy.

Let's start with the policy difference mentioned in last post.

Let $$ \eta(\pi) = \mathbb{E}_{s_0,a_0,...}{\Large[}\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t){\Large]}$$ where
$$s_0 \sim \rho_0(s_0), a_t \sim \pi(a_t|s_t), s_{t+1} \sim P(s_{t+1}|s_t,a_t)$$

Then the difference of expected reward between policies is
$$\eta(\tilde{\pi}) - \eta(\pi) = \mathbb{E}_{s_0,a_0,... \sim \tilde{\pi}}\sum_{t=0}^{\infty} \gamma^t {\large(}Q^{\pi}(s_t,a_t) - V^{\pi}(s_t){\Large)}$$

Here we can define Advantage funciton as
$$A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)$$

And we can define state distribution as
$$\rho_{\pi}(s)=P(s_0=s) + \gamma P(s_1=s) + \gamma^2 P(s_2=s) + ...$$

Then, $$\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{s_0,a_0,... \sim \tilde{\pi}}\sum_{t=0}^{\infty} \gamma^t A^{\pi}(s_t,a_t)$$
$$ = \eta(\pi) + \sum_{t=0}^{\infty} \sum_s P(s_t=s|\tilde{\pi})\sum_a\tilde{\pi}(a|s)\gamma^t A_{\pi}(s,a)$$
$$ = \eta(\pi) + \sum_{t=0}^{\infty} \sum_s \gamma^t P(s_t=s|\tilde{\pi})\sum_a\tilde{\pi}(a|s)\gamma^t A_{\pi}(s,a)$$
$$ = \eta(\pi) + \sum_s \rho_{\tilde{\pi}}(s) \sum_a \tilde{\pi}(a|s) \gamma^t A_\pi(s,a) $$

From this equation we can imply that by allocating 
$$\tilde{\pi}(a|s)$$ to non-negative Advantage functions $${\Large(} A_\pi(s,a) > 0 {\Large)}$$ at each state, we can find improved (at least same) policy. However, estimation and approximation error exists. And since we don't have $$\rho_{\tilde{\pi}}(s)$$, it is hard to optimize $$\eta(\tilde{\pi})$$ directly.

Instead, we introduce the local approximation of $$\eta$$:

$$L_{\pi}(\tilde\pi) = \eta(\pi) + \sum_s \rho_\pi (s) \sum_a \tilde{\pi} (a|s) A_\pi(s,a)$$

This local approximation uses state distribution of current policy $$\pi$$ rather than target policy $$\tilde{\pi}$$. So it does not consider state distribution change due to policy change, which could be valid if two policies are not different much.
And $$L_\pi$$ is called local approximation because it matches $$\eta$$ to first order.

$$ L_{\pi_{\theta_0}} = \eta(\pi_{\theta_0}),$$

$$ \nabla _\theta L_{\pi_{\theta_0}}|_{\theta=\theta_0} = \nabla_\theta \eta(\pi_\theta)|_{\theta=\theta_0}$$

So sufficiently small step of gradient that optimizes $$L_{\pi_{\theta_{old}}}$$ will also improve $$\eta$$ but how much is the problem.

To address this issue, Kakade & Langford proposed a lower bound.<br> 
Let $$\pi'= argmax_{\pi'} L_{\pi_{old}}(\pi')$$, Then for mixture policy

$$\pi_{new} = (1-\alpha)\pi_{old}(a|s) + \alpha\pi'(a|s) $$

the following lower bound holds

$$\eta(\pi_{new}) \ge L_{\pi_{old}}(\pi_{new}) - \frac{2\epsilon\gamma}{(1-\gamma)^2}\alpha^2$$

where, 
$$\epsilon = max_s|\mathbb{E}_{a\sim \pi'(a|s)}[A_\pi(s,a)]|$$

<br>
By iteratively forming lower bound and optimizing it, we can continuously improve the policy. But the problem is that the mixer policy is hard to use for general stochastic policies.

So the author extended this to general stochastic policy using total variation divergence which is defined as
$$ D_{TV}(p||q) = \frac{1}{2}\sum_i|p_i-q_i|$$, 
where p and q are discrete probability distributions.<br>
Using this divergence, we can replace original alpha with $$\alpha = D_{TV}^{max}(\pi_{old},\pi_{new}) = max_s D_{TV}(\pi_{old}(\cdot|s)||\pi_{new}(\cdot|s))$$.

Using this alpha, following lower bound again holds.

$$\eta(\pi_{new}) \ge L_{\pi_{old}}(\pi_{new}) - \frac{2\epsilon\gamma}{(1-\gamma)^2}\alpha^2$$

where, 
$$\epsilon = max_{s,a}|A_\pi(s,a)|$$

And according to Pollard, the following realtionship holds between total variation divergence and the KL divergence

$$D_{TV}(p,q)^2 \ge D_{KL}(p||q)$$

From this, let 
$$ D_{KL}^{max}(\pi,\tilde\pi) = max_s D_{KL}(\pi(\cdot|s) || \tilde\pi(\cdot|s)).$$

Then, the following bound holds

$$ \eta(\tilde\pi) \ge L_\pi(\tilde\pi) - \frac{2\epsilon\gamma}{(1-\gamma)^2}D_{KL}^{max}(\pi,\tilde\pi)$$

Again, using this lower bound, we can optimize our policy. But in practice, due to the large value of $$\frac{2\epsilon\gamma}{(1-\gamma)^2}$$, the size of optimization steps taken become very small (since $$\gamma \approx 1 $$). In order to take larger steps, one way is to transform our KL-divergence term into the constraint (trust region constraint).<br>
By turning KL-divergence to constraint, our optimization function becomes

$$ maximize_{\theta} \;L_{\theta_{old}}(\theta) $$

$$ subject\,to \; D_{KL}^{max}(\theta_{old},\theta)\le\delta$$

This transformation is valid because transformed function is a lower bound of equivalent function of original function.

Suppose $$ D_{KL}^{max}(\theta_{old},\theta) \le \delta $$  holds.<br>
Then it is clear that $$L_{\theta_{old}}(\theta) \le L_{\theta_{old}}(\theta) - \frac{2\epsilon\gamma}{(1-\gamma)^2}(D_{KL}^{max}(\theta_{old},\theta) - \delta)$$

Since right side of inequality is equivalent function of original lower bound, we can use $$L_{\theta_{old}}(\theta)$$ as a lower bound of original lower bound. And by using this, we can take a larger optimization step which could be helpful for our optimizing process.

In practice, it is almost impossible to check KL-divergence at every states to measure $$D_{KL}^{max}$$. Author explained in paper that even if we use average KL-divergence instead of maximum KL-divergence, we can get similar performance empirically.

We can define average KL-divergence as

$$\overline{D}_{KL}^{\rho}(\theta_1,\theta_2) = \mathbb{E}_{s \sim \rho}[D_{KL}(\pi_{\theta_1}(\cdot|s)||\pi_{\theta_2}(\cdot|s))]$$

So the final optimization problem becomes

$$maximize_\theta \; L_{\theta_{old}}(\theta)$$

$$subject\,to\; \overline{D}_{KL}^{\rho_{\theta_{old}}}(\theta_{old},\theta) \le \delta $$

<br><br><br>


<h1>Practical Implementation</h1>
<br>
So what we want to mazimize is

$$ maximize_{\theta} \; \sum_s\rho_{\theta_{old}}(s)\sum_a\pi_\theta(a|s)A_{\theta_{old}}(s,a) $$

$$subject\,to\; \overline{D}_{KL}^{\rho_{\theta_{old}}}(\theta_{old},\theta) \le \delta $$

$$\sum_s \rho_{\theta_{old}}(s)$$is changed to $$\frac{1}{1-\gamma}\mathbb{E}_{s\sim\rho_{\theta_{old}}}$$ since we are sampling states following $$\pi_{\theta_{old}}. $$                         

For $$A_{\theta_{old}}(s,a) $$
, we use $$\hat{Q}_{\theta_{old}}(s,a)$$ instead. This is because we estimate value function with rollout result.<br> Since $$\sum_a\pi_\theta(a|s)V_{\theta_{old}}(s,a)$$ is constant with respect to $$\theta$$, it is okay to replace $$A_{\theta_{old}}(s,a)$$ with $$Q_{\theta_{old}}(s,a)$$.

For sampling action, we use action sampling distribution 
$$q(a|s)$$.<br> For samll finite action spaces, we can use uniform distribution and consider every actions. For large or continuous state spaces, we usually sample with $$q(a|s)\,=\,\pi_{\theta_{old}}(a|s)$$ and compensate this with importance sampling.

Now our optimization problem becomes

$$maximize_\theta\;\mathbb{E}_{s\sim\rho_{\theta_{old}},a\sim q}{\Large[}\frac{\pi_\theta(a|s)}{q(a|s)}\hat{Q}_{\theta_{old}}(s,a){\Large]}$$

$$subject\,to\; \mathbb{E}_{s \sim \rho_{\theta_{old}}}[{D}_{KL}(\pi_{\theta_{old}}(\cdot|s),\pi_\theta(\cdot|s))] \le \delta $$

Solving this optimization problem is similar to natural policy grdaient.

We use a linear approximation of $$L_{\theta_{old}}(\theta)$$ and a quadratic approximation of $$\overline{D}_{KL}$$ constraint.

Here, KL-divergence is approximated using Fisher information matrix $$A$$.<br>

Let $$A(\theta_{old})_{ij} = \frac{\delta}{\delta\theta_j}\frac{\delta}{\delta\theta_i}\mathbb{E}_{s \sim \rho_\pi}[D_{KL}(\pi(\cdot|s,\theta_{old})||\pi(\cdot|s,\theta))]|_{\theta=\theta_{old}}\;$$
then 
$$\overline{D}_{KL}^{\rho_{\theta_{old}}}(\theta_{old},\theta) \approx (\theta-\theta_{old})A(\theta_{old})(\theta-\theta_{old})^T $$

<br>
Let,
$$ g \,=\, \nabla_\theta(L_{\theta_{old}})|_{\theta=\theta_{old}} $$
then $$L_{\theta_{old}}(\theta) \approx g(\theta-\theta_{old})$$

Then the search direction is $$s = A^{-1}g\,$$ and the step size should be the maximum value that satisfies the KL-divergence constraint and improves surrogate objective function.

Measuring fisher information matrix and Calculating its inverse is not easy. Thus Trust Region Policy Optimization is known to be computationally expensive.


<br><br>

<h1>References</h1>
Trust Region Policy Optimization, Schulman et al. 2015
<a href="https://arxiv.org/abs/1502.05477">https://arxiv.org/abs/1502.05477</a>

