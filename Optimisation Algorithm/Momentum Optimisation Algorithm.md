SGD faces issues while performing traversal when there exists non-convex curves of the loss function:

1. Local Minima: If the SGD attains a local minima instead of a Global minima, the algorithm cannot move forward. It assumes that the local minima is the best fit as the slope is nearly zero. This is often seen in noisy curvature of loss function.

2. Saddle Point: Saddle point is where the loss function has a minima at one side and maxima at the other. Here the steepness of the slope is small which makes SGD to move slowly.

### Solving issues of SGD - Speed
Most of the issues faced by SGD is due to slow decrease in gradient. This causes the gradient to get stuck in local minima or at the saddle point. The issue can be solved by increasing the speed at which the gradient updates the weights. If the weights are increased in a faster momentum. This is introduced to the algorithm using parameter $\beta$ called as **decay factor**. 

To establish momentum in Gradient Descent is add the change in the weights by performing Gradient Descent on it. It is given as :$\alpha~*~gradient~+~\beta~*~change~in~weights$ 

![[momentum_sgd.png]]
### Updating weights
The weights updates using Momentum Algorithm is given by:
$v_{t+1} = \beta v_t + (1 - \beta) \nabla f(w_t)$
$w_{t+1} = w_t - \alpha v_{t+1}$

$where:$
$v_t \text{ is the current momentum}$
$v_{t+1} \text{ is updated momentum}$
$\beta \text{ is the momentum coefficient}$
$\nabla f(w_t) \text{ is the gradient of the loss function with respect to the weights}$
$w_t \text{ is the current weight}$
$w_{t+1} \text{ is the updated weight}$ 
$\alpha \text{ is the learning rate}$





