# Deep Q Learning

### Q Learning
> *Algorithm predicts the value of a state-action pair, and then you compare this prediction to the observed accumulated rewards at some later time and update the parameters of your algorithm optimising it*

$$Q(S_t, A_t) = Q(S_t, A_t) \ + \ \alpha [R_{t+1} \ + \ \gamma \ max\ Q(S_{t+1})\ - \ Q(S_t, A_t) ]$$

