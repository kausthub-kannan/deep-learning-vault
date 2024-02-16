# Multi Arm Bandit Problem  

#### Problem Statement with No Time Constraint:  
A casino game house with *n* casino games. Each game provides rewards with a probabilistic distribution which can be played for no cost. The slot machines have one lever each for performing the action, hence a total of *n* levers. Formally the problem is re-worded as:  

>  ***n** possible actions where an action means pulling the lever of a particular slot machine, and at each play **k** of this game we can choose a single lever to pull. After taking an action (a) we will receive a reward, **R** $_k$ (reward at play k). Each lever has a unique probability distribution of payouts (rewards)*  
#### Greedy Approach  

Concept of **Expected Reward** (Q$_k$) is used in the process. For every action an expected reward is returned.  

```python
def get_action_value(action, history=history):  
	rewards_for_action = history[action]  
	return sum(rewards_for_action) / len(rewards_for_action)
```

As the expected award is the mean of all previous rewards for action *a*, there is influence of previous actions on current and future actions (time as dimension along with space).   

RL and deep learning uses two terms; **Exploration** and **Exploitation**.   
Exploration is where random choice of slot machine is taken and reward is obtained. Exploitation is where with prior information about the game (environment) choice is done. Initially exploration is done to gain information but soon exploitation is take up such that we maximise the reward.   

This problem can be solved with a simple  novice maximisation code as:  $A_t = argmax \ Q_t(a)$
```python
def get_best_action(actions):    
	best_action = 0    
	max_action_value = 0    
	for i in range(len(actions)):              
		cur_action_value = get_action_value(actions[i])            
		if cur_action_value > max_action_value:            
			best_action = i            
			max_action_value = cur_action_value    
	return best_action
```


**Epsilon-Greedy approach:**  
The problem with the previous case is the lack of exploration of every bandit. We explore each bandit once and hence unable to know the true possible profit as the distribution of each bandit remains unknown.   

To do do more exploration we use a threshold value $\epsilon$. Till this value, continuous exploration of every bandit is done after which with the obtained mean of every bandit ($\mu_k$), the best is chosen. This provides some balance to exploration vs exploitation trade off.  

**Note:**
In case of assigning rewards, if the probability of say choice *c* is greater than others (say 9 bandits out of 10), it would be given reward each time it is better than other bandit.

#### Softmax Approach:
Unlike previous approaches, Softmax provides probability distribution for the bandit. Hence instead of choosing the bandit randomly, every time we chose a bandit and obtain reward it's probability is obtained. Next we can chose bandits who have better probability obtain in set of previous actions.  

```python
def update_record(record,action,r): #Update Mean reward for every choice 
	new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1) 
	record[action,0] += 1    
	record[action,1] = new_r    
	return record
	
def get_reward(prob, n=10):    
	reward = 0    
	for i in range(n):        
		if random.random() < prob: #Explore other possibilities else exploit the best arm if our probability (can be epsilon) is lesser than random probability.
			reward += 1    
	return reward
	
def softmax(av, tau=1.12):    
	softm = np.exp(av / tau) / np.sum( np.exp(av / tau) )     
	return softm

rewards = [0] 
for i in range(500):    
	p = softmax(record[:,1])            
	choice = np.random.choice(np.arange(n),p=p) # Weights(p) recived from softmax
	r = get_reward(probs[choice]) # Sending softmax probability insted of epsilon
	record = update_record(record,choice,r)    
	mean_reward = ((i+1) * rewards[-1] + r)/(i+2)    
	rewards.append(mean_reward)
```

