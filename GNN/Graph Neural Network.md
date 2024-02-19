# Graph Neural Network

GNNs are derived versions of Convolution but for non-Euclidean data. The defining feature of GNN is the **Neural Message Passing** of vector messages and are exchanged between nodes. 

#### Neural Message Passing
Message passing works on the basics of collecting the features *X* from the neighbor embedding and then combined to form a new hidden embedding for the node. 

![[Neural Message Passing.png]]

Let $h^{(k)}_u$ be the hidden embedding of node $u$ at the $kth$ iteration.  The change in the hidden embedding follows the following steps:
1. For every neighboring node $\nu$, $h^{(k)}_\nu$ is fetched. 
2. Aggregation of the fetched $h^{(k)}_\nu$ is done.
3. Now the $h^{(k)}_u$ is updated with the aggregated embedding.
$$h^{(k)} = UPDATE^{(k)} (h^{k}_u, AGGREGATE(h^{(k)}_\nu))$$

The $AGGREGATE(h^{(k)}_\nu)$ is called as the message and is represented by $m^{(k)}_{N(u)}$.  The 

#### Basic GNN
Basic GNN can be represented as: $$h^{(k)}_u = \sigma (W^{(k)}_{self} h^{(k-1)}_u) \ + \ W^{(k)}_{negih} \Sigma \ h^{(k)}_\nu \ + \ b^{(k)}$$
where: $W$ is trainable weights with $b$ being the bias

GNN uses message passing but with variations in how the Update and Aggregation is performed. Few major ones are:
1. Self Loops and Normalization: $h^{(k)}_u = \sigma (W^{(k)}_{self} \Sigma \frac{h^{(k-1)}_u)}{\sqrt{|N(u)||N(\nu)|}})$
2. Aggregation via MLP: $m_{N(u)} = MLP_\theta(\Sigma MLP_\phi (h_\nu))$
3. Aggregation via Attention: $m_{N(u)} = \Sigma \alpha_{u,\nu}h_\nu$
4. RNN Approach: $h^{(k)}_u = GRU(h^(k-1)_u)$