Gradient Descent Algorithms uses the approach of Optimisation using partial derivatives. 

**Gradient:**
It is defined a the matrix of partial derivatives for given *n* features. For example for a two feature data gradient can be represented as: $[\frac{\delta f}{\delta x} \frac{\delta f}{\delta y}]^T$ 

Gradient Descent utilises gradient to find the global minima of the given function to be optimised upon. For a minima, Gradient is equal to zero. This provides the slope or tangent of the initial point which has be taken by the model. The steepness of the slope is taken into consideration. The weights and basis are updated only if the slope is negative. To reach the minima, the weights will be updated according to the **Loss Function**. Iteration will continue until minima is reached which occurs when the loss is close to zero.

#### Loss Function:
Loss function measures the difference, or error, between true target and predicted target variables. This improves the machine learning model's efficacy by providing feedback to the model so that it can adjust the parameters to minimize the error and find the local or global minimum. This is termed as **Backward Propagation** or **Backward Pass** which is used to calculate the loss and be fed for optimisation. 

#### Learning Rate:
It is the size of the steps that are taken to reach the minimum. This is typically a small value, and it is evaluated and updated based on the behaviour of the cost function. High learning rates result in larger steps but risks overshooting the minimum. Conversely, a low learning rate has small step sizes. While it has the advantage of more precision, the number of iterations compromises overall efficiency as this takes more time and computations to reach the minimum.

![[learning_rate.png]]

### Issues with Gradient Descent:
1. **Vanishing Gradient:**  This occurs when the gradient is too small. As we move backwards during backpropagation, the gradient continues to become smaller, causing the earlier layers in the network to learn more slowly than later layers. When this happens, the weight parameters update until they become insignificant—i.e. 0—resulting in an algorithm that is no longer learning. Often activation functions such as Leaky ReLU helps in avoiding such issues. This issue is often seen in RNN networks and GAN models.

2. **Exploding Gradient:** This happens when the gradient is too large, creating an unstable model. In this case, the model weights will grow too large, and they will eventually be represented as NaN. One solution to this issue is to leverage a dimensional reduction technique, which can help to minimize complexity within the model.

![[local_minima_and_saddle_point.png]]

## SGD - Stochastic Gradient Descent

SGD is based on the principle of Gradient Descent. The additional aspect of SGD is that huge dataset can be broken in several batches is taken. To each batch the training and optimisation is done. The key feature of SGD is that it takes in *random batch or data instance* by doing so introduces *stochastic* or random characters to the model parameters. Along with randomness as SGD trains upon iteration of small batches, unlike traditional Gradient Descent it reduces computational cost as the complete huge dataset need not be loaded in a single shot.  Yet it is not as computationally efficient as the Batch Gradient Descent but covers it up with more detail and accurate weight updates and increased speed. 

The formula for the weights getting updated is given by: 
$w_{t+1} = w_t - \alpha \nabla f(w_t)$
$where:$ 
$w_t ~\text{is current weight}$ 
$w_{t+1}~\text{is updated weight}$
$\alpha~\text{is learning rate}$
$\Delta f(w_t)~\text{is gradient of loss function}$ 




