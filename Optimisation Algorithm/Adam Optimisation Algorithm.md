Adaptive Moment Estimation (ADAM) optimiser is an extension of SGD. Unlike SGD though, Adam adaptively updates learning rate. Adam optimiser dynamically computes individual learning rates based on the past gradients and their second moments. It is a merge of Momentum and adaptive learning rate technique. 

The past gradient approach used in:
Momentum - $v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla f(w_t))^2$
RMSProp - $m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla f(w_t)$

Adam combines both these approaches. This makes Adam capable of handling local minima problem.
### Updating weights
$m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla f(w_t)$
$v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla f(w_t))^2$
$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}$
$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$
$w_{t+1} = w_t - \frac{\alpha}{\sqrt{\hat{v}_{t+1} + \epsilon}} \hat{m}_{t+1}$

$\text{where}:$
$w_t ~\text{is current weight}$ 
$w_{t+1}~\text{is updated weight}$
$\alpha~\text{is learning rate}$
$\Delta f(w_t)~\text{is gradient of loss function}$ 
$m_t~and~v_t~\text{are the first and second moments of the gradients}$
$\hat{m}_{t+1}~\text{and}~\hat{v}_{t+1}~\text{are bias-corrected estimates}$ 
$\epsilon~\text{is a small constant added for numerical stability}$



