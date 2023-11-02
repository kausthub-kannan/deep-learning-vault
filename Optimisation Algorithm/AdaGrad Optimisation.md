AdaGrad adopts the learning rate which gives its name. In data given if features are sparse .i.e if most of the values are 0 then AdaGrad is suitable. Sparse data results in loss function curve to be elongated (the cross-section of global minima is elliptical when compared to circular). Hence slope of on axis has very small steepness compared to other. This elongated cure is called **valley**. 

![[adagrad.png]]

The spare features results in very small update in weights and biases slowly leading to vanishing gradient with the existence of constant learning rate. To solve this AdaGrad performs different learning rates which are adaptive. This reduces the learning rate size hence increasing the overall gradient size.

The adaptive capability of AdaGrad is bought by dividing the learning rate by the previous gradient. By doing to if the previous gradient is small, the learning rate would not change a lot but the gradient is lot then the learning rate is reduced.
### Updating weights
Weights are updated as below (same goes for biases):

$v_{t+1} = v_t + (\nabla f(w_t))^2$
$w_{t+1} = w_t - \frac{\alpha}{\sqrt{v_{t+1} + \epsilon}}$

$w_t ~\text{is current weight}$ 
$w_{t+1}~\text{is updated weight}$
$\alpha~\text{is learning rate}$
$\Delta f(w_t)~\text{is gradient of loss function}$ 
$v_t ~\text{is previous gradient sums}$
$v_t+1 ~\text{is updated gradient sum}$
$\epsilon ~\text{added for numerical stability}$



