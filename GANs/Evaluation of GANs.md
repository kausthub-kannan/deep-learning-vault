# Evaluation of GANs

The key elements evaluating generated images are:
1. **Fidelity**
2. **Diversity**

### Fréchet Inception Distance:
A state-of-art evaluation technique for GANs which based on Feature extraction. Here features are compared which more reliable than Pixel distance. For this the features have to be extracted and the compared. In classifiers, the fully connected layer is removed and the pooling layers gives features which condensed than the actual output. It compares the normal distributions of Real (x) and fake (y) embedding.

$\text{FID} = \|\mu_x - \mu_y\|^2 + \text{Tr}(\Sigma_x + \Sigma_y - 2(\Sigma_x\Sigma_y)^{1/2})$
$where:$  
$\mu_x \ \text{mean real image distribution}$  
$\mu_y \ \text{mean fake image distribution}$   
$\Sigma_x \ and \ \Sigma_y \ \text{are covariance matrix of real and fake distribtuions}$

### Perpetual Path Length  
PPL correlates with consistency and stability of shapes and was introduced with StyleGAN. It has two division to measure w-space and z-space called linear and spherical interpolation respectively.  

### Inception Score
Inception Score works on Feature extraction on Inception V3 embedding This uses KL Divergence in exponential scale to analysis generated images and evaluate them.  
***Working:***  
KL Divergence provides analysis of Conditional probability and Marginal Probability which gives Fidelity and Diversity of the generated image distribution.

***Limitations:***
1. Does not avoid mode collapse  
2. No comparison to real image distribution  
3. Inception V3 trained on ImageNet does not provide complete diversity

### Sampling for Evaluation
Sampling is key aspect in evaluating GANs as out feature extraction evaluation methods evaluate on sampled results. The sample images have to be diverse and high fidelity. This ensures we cover the required important metrics of GANs.    
  
Truncation ensures that there is right compromise in fidelity and diversity. The tails of the distribution provides high diversity and the peaks ensure high fidelity. Truncation can be highly parameterised and hence giving control over evaluating GANs. It trims tails and shortens amplitude of distribution.  

### Precision and Recall

#### Precision:  
1. Focuses on fidelity of the generated images
2. In the distribution overlap of the generated and real image distribution, precision focuses on the how much of generated distribution is not overlapping with the real distribution
3. It is the ratio of overlap distribution fakes to the total number of generated images.

### Recall:  
1. Focuses on diversity of the generated images
2. It focuses on the non-overlap part of the real distribution with the generated distribution
3. Recall speaks about how much of images the generator couldn't model.
4. It is the ratio of overlap distribution fakes to the total number of real images
