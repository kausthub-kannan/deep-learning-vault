# Deep Q Learning

> *Q Learning is an algorithm **predicts the value of a state-action pair**, and then you compare this prediction to the observed accumulated rewards at some later time and update the parameters of your algorithm to optimise itself.*

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot ( r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a))$$
```python
from Gridworld import Gridworld # Import py file making the env
import numpy as np
import random from matplotlib 
import pylab as plt
import torch
import torch.nn as nn

game = Gridworld(size=4, mode='static')

l1 = 64 
l2 = 150 
l3 = 100 
l4 = 4 
model = torch.nn.Sequential(    
							torch.nn.Linear(l1, l2),    
							torch.nn.ReLU(),    
							torch.nn.Linear(l2, l3),    
							torch.nn.ReLU(),    
							torch.nn.Linear(l3,l4) 
							) 
loss_fn = torch.nn.MSELoss() 
learning_rate = 1e-3 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
gamma = 0.9 
epsilon = 1.0
action_set = { 0: 'u', 1: 'd', 2: 'l', 3: 'r', }
```

Training:
```python
epochs = 1000 
losses = []

for i in range(epochs):                            
	game = Gridworld(size=4, mode='static')            
	state_ = game.board.render_np().reshape(1,64)+ np.random.rand(1,64)/10.0         state1 = torch.from_numpy(state_).float()        
	status = 1        

	while(status == 1):  # play the whole game until it's win or loss                                 
		qval = model(state1) # predict action-state pair (to be sent to loss)
		qval_ = qval.data.numpy()   
		     
		if (random.random() < epsilon): # random action selection            
			action_ = np.random.randint(0,4)        
		else:            
			action_ = np.argmax(qval_)

		 action = action_set[action_] # select an action from the set         
		 game.makeMove(action) # perform the move                                
		 state2_=game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
		 state2 = torch.from_numpy(state2_).float() # new state
		 reward = game.reward() # get the reward for performed predicted action  
		 
		 with torch.no_grad():            
			 newQ = model(state2.reshape(1,64)) # Run for new state for maxQ     
		maxQ = torch.max(newQ)       

		 if reward == -1: # If reward = -1, the game is not over                 
			 Y = reward + (gamma * maxQ) # Target Value for loss calculation 
		else:            
			Y = reward        
			
		y_true = torch.Tensor([Y]).detach()        
		y_pred = qval.squeeze()[action_] 
		loss = loss_fn(y_pred, y_true)
		        
		optimizer.zero_grad()        
		loss.backward()    
		losses.append(loss.item())        
		optimizer.step()        
		state1 = state2

		if reward != -1:                
			status = 0    
		if epsilon > 0.1:                 
			epsilon -= (1/epochs)
```

