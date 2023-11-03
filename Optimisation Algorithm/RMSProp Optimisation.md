# RMSProp Optimisation  

In AdaGrad, the updates of gradients are really small. This makes AdaGrad to never converge with the minima. This is solved by making the past gradient sum to decay on exponential basis using a small parameter $\beta$. 

### Updating weights
$s_{t+1} = \beta s_t + (1 - \beta) (\nabla f(w_t))^2$  
$w_{t+1} = w_t - \frac{\alpha}{\sqrt{s_{t+1} + \epsilon}}$  

$\text{where:}$
$w_t \ \text{is current weight}$   
$w_{t+1} \ \text{is updated weight}$
$\alpha \ \text{is learning rate}$  
$\Delta f(w_t) \ \text{is gradient of loss function}$   
$s_t \ \text{is moving average of squared gradients}$  
$\epsilon \ \text{added for numerical stability}$  
$\beta \ \text{is for decaying past gradient sum}$  






