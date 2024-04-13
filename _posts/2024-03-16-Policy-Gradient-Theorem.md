---
layout: post
title: Policy Gradient Theorem
use_math: true
summary: Policy Gradient Theorem for discounted reward setting.
---
<head>

</head>

<h1 style = "color:#d28445;font-weight:bold">
Policy Gradient Theorem for discounted Episodic case
</h1>
<br>
<main class = "content">
Instead of learning value functions and using it to action selection (e.g. $ \epsilon$ greedy) we can learn parameterized policy and select action without consulting value function. Policy gradient theorem is the fundamental theorem which makes possible to learn parameterized policy.

Here we consider episodic, discounted case with initial state s.<br><br> We want to learn policy that maximizes

$$\begin{array}{l}
 V^{\pi}(s) = \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t)\right] 
\end{array}$$

using $\nabla_{\theta} V^{\pi_\theta}(s)$
<br>
($\tau$ is a sample trajectory when following policy $\pi$)
<br><br><br>
Performance difference between policies can be defined as

$$\begin{array}{l}
V^{\pi}(s) - V^{\pi'}(s) = \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}{\Large[}\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t){\Large]} - V^{\pi'}(s)\\
= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + V^{\pi'}(s_t) - V^{\pi'}(s_t){\Large)}\right] - V^{\pi'}(s)\\
= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + \gamma V^{\pi'}(s_{t+1}) - V^{\pi'}(s_t){\Large)}\right]\\
= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + \gamma V^{\pi'}(s_{t+1}) - V^{\pi'}(s_t){\Large)}\right]\\
= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + \gamma \mathbb{E}[V^{\pi'}(s_{t+1})|s_t,a_t] - V^{\pi'}(s_t){\Large)}\right]\\
= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(} Q^{\pi'}(s_t,a_t) - V^{\pi'}(s_t){\Large)}\right]
\end{array}$$
<br>
<p>
Now, consider $ Pr^\pi(s_t=s'|s_0=s) $: probability of reaching state $s'$ in time $t$ starting at state $s$.
</p>

We can define state distribution as

$$\begin{array}{l}
d_s^\pi(s') = {(1-\gamma)}\sum_{t=0}^{\infty}\gamma^{t}Pr^\pi(s_t=s'|s_0=s)
\end{array}$$

Using this, we can define policies performance difference as

<span class="math">
\begin{array}{l}
V^{\pi}(s) - V^{\pi'}(s) = \frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\mathbb{E}_{a\sim\pi(a|s')}\left[{\Large(}Q^{\pi'}(s',a) - V^{\pi'}(s'){\Large)}\right]\\
=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\left[\sum_{a}{\Large(}\pi(a|s')Q^{\pi'}(s',a){\Large)} - V^{\pi'}(s')\right]\\
=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\left[\sum_{a}{\Large(}\pi(a|s')Q^{\pi'}(s',a) - \pi'(a|s')Q^{\pi'}(s'){\Large)}\right]\\
=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\left[\sum_{a}{\LARGE(}{\Large(}\pi(a|s')-\pi'(a|s'){\Large)}Q^{\pi'}(s',a){\LARGE)}\right]
\end{array}
</span>

From this we can show that

$$\begin{array}{l}
\nabla_{\theta} V^{\pi_\theta}(s) = \lim_{\epsilon \to 0} \frac{V^{\pi_{\theta +\epsilon}}(s) - V^{\pi_\theta}(s)}{\epsilon}\\
=\lim_{\epsilon \to 0}\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta} }\left[\sum_{a}{\LARGE(}\frac {\pi_{\theta+\epsilon} (a|s')-\pi_\theta (a|s')}{\epsilon}Q^{\pi_\theta}(s',a){\LARGE)}\right]\\
=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta}}\sum_{a}\nabla\pi_\theta(a|s')Q^{\pi_\theta}(s',a)\\
=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta}}\sum_{a}\frac {\nabla\pi_\theta(a|s')} {\pi_\theta(a|s')} {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)\\
=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta}}\mathbb{E}_{a\sim \pi_\theta(a|s')}\nabla_\theta \log {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)
\end{array}$$

We can generalize this result to arbitrary initial state distribution $s \sim \mu(s)$.

Hence<br><br>

$\begin{array}{l}
\nabla_\theta J(\theta)=(1-\gamma)\mathbb{E}_{s\sim \mu(s)}\nabla_\theta V^{\pi_\theta}(s)
=\mathbb{E}_{s'\sim d_\mu^{\pi_\theta}}\mathbb{E}_{a\sim \pi_\theta(a|s')}\nabla_\theta \log {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)
\end{array}$
<br>

where $J(\theta)$ is a policy performance function of policy parameter $\theta$. 
</main>
