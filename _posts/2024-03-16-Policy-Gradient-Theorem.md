---
layout: post
title: Policy Gradient Theorem
summary: Policy Gradient Theorem for discounted reward setting.
---
<head>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
</head>


**<span style="color:orange;font-size:200%">Policy Gradient Theorem for discounted Episodic case</span>**

<span style="font-size:150%"> Instead of learning value functions and using it to action selection (e.g. $\epsilon$ greedy) we can learn parameterized policy and select action without consulting value function.<br>
 Policy gradient theorem is the fundamental theorem which makes possible to learn parameterized policy.<br><br>
 Here we consider episodic, discounted case with initial state s.<br> We want to learn policy that maximizes<br>
 $$V^{\pi}(s) = \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t)\right]$$<br>
 using $\nabla_{\theta} V^{\pi_\theta}(s)$<br><br>
 ($\tau$ is a sample trajectory when following policy $\pi$)<br><br>
 Performance difference between policies is<br>
 $$V^{\pi}(s) - V^{\pi'}(s) = \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}{\Large[}\sum_{t=0}^{\infty}\gamma^tr(s_t,a_t){\Large]} - V^{\pi'}(s)$$<br>
 $$= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + V^{\pi'}(s_t) - V^{\pi'}(s_t){\Large)}\right] - V^{\pi'}(s)$$<br>
 $$= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + \gamma V^{\pi'}(s_{t+1}) - V^{\pi'}(s_t){\Large)}\right]$$<br>
 $$= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + \gamma V^{\pi'}(s_{t+1}) - V^{\pi'}(s_t){\Large)}\right]$$<br>
 $$= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(}r(s_t,a_t) + \gamma \mathbb{E}[V^{\pi'}(s_{t+1})|s_t,a_t] - V^{\pi'}(s_t){\Large)}\right]$$<br>
	$$= \mathbb{E}_{\tau \sim Pr^\pi(\tau|s_o=s)}\left[\sum_{t=0}^{\infty}\gamma^t{\Large(} Q^{\pi'}(s_t,a_t) - V^{\pi'}(s_t){\Large)}\right]$$<br><br>
 Now, consider $Pr^\pi(s_t=s'|s_0=s)$: probability of reaching state $s'$ in time $t$ starting at state s.<br>
 We can define state distribution as
 $$d_s^\pi(s') = {(1-\gamma)}\sum_{t=0}^{\infty}\gamma^{t}Pr^\pi(s_t=s'|s_0=s)$$<br><br>
 Using this, we can define policies performance difference as
 $$V^{\pi}(s) - V^{\pi'}(s) = \frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\mathbb{E}_{a\sim\pi(a|s')}\left[{\Large(}Q^{\pi'}(s',a) - V^{\pi'}(s'){\Large)}\right]$$<br>
 $$=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\left[\sum_{a}{\Large(}\pi(a|s')Q^{\pi'}(s',a){\Large)} - V^{\pi'}(s')\right]$$<br>
 $$=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\left[\sum_{a}{\Large(}\pi(a|s')Q^{\pi'}(s',a) - \pi'(a|s')Q^{\pi'}(s'){\Large)}\right]$$<br>
 $$=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^\pi}\left[\sum_{a}{\LARGE(}{\Large(}\pi(a|s')-\pi'(a|s'){\Large)}Q^{\pi'}(s',a){\LARGE)}\right]$$<br><br>
 From this we can show that<br>
 $$\nabla_{\theta} V^{\pi_\theta}(s) = \lim_{\epsilon \to 0} \frac{V^{\pi_{\theta +\epsilon}}(s) - V^{\pi_\theta}(s)}{\epsilon}$$
 $$=\lim_{\epsilon \to 0}\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta} }\left[\sum_{a}{\LARGE(}\frac {\pi_{\theta+\epsilon} (a|s')-\pi_\theta (a|s')}{\epsilon}Q^{\pi_\theta}(s',a){\LARGE)}\right]$$<br>
 $$=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta}}\sum_{a}\nabla\pi_\theta(a|s')Q^{\pi_\theta}(s',a)$$<br><br>
 $$=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta}}\sum_{a}\frac {\nabla\pi_\theta(a|s')} {\pi_\theta(a|s')} {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)$$<br><br>
 $$=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_s^{\pi_\theta}}\mathbb{E}_{a\sim \pi_\theta(a|s')}\nabla_\theta \log {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)$$<br><br>
 We can generalize this result to arbitrary initial state distribution $s \sim \mu(s)$<br><br>
 Hence it can be written as
 $$\nabla_\theta J(\theta)=\frac{1}{1-\gamma}\mathbb{E}_{s'\sim d_\mu^{\pi_\theta}}\mathbb{E}_{a\sim \pi_\theta(a|s')}\nabla_\theta \log {\pi_\theta(a|s')} Q^{\pi_\theta}(s',a)$$<br>
 where $J(\theta)$ is a policy performance function of policy parameter $\theta$. 
</span>

