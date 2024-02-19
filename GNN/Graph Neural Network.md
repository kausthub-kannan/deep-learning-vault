# Graph Neural Network

GNNs are derived versions of Convolution but for non-Euclidean data. The defining feature of GNN is the **Neural Message Passing** of vector messages and are exchanged between nodes. 

#### Neural Message Passing
Message passing works on the basics of collecting the features *X* from the neighbor embedding and then combined to form a new hidden embedding for the node. 

![[Pasted image 20240219200400.png]]

Let $h^{(k)}_u$ be the hidden embedding of node $u$ at the $kth$ iteration.  The change in the hidden embedding follows the following steps:
1. For every neighboring node $\nu$, $h^{(k)}_\nu$ is fetched. 
2. Aggregation of the fetched $h^{(k)}_\nu$ is done.
3. Now the $h^{(k)}_u$ is updated with the aggregated embedding.
$$h^{(k)} = UPDATE^{(k)} (h^{k}_u, AGGREGATE(h^{(k)}_\nu))$$

The $AGGREGATE(h^{(k)}_\nu)$ is called as the message and is represented by $m^{(k)}_{N(u)}$.  The Basic GNN can be represented as: $$h^{(k)}_u = \sigma (W^{(k)}_{self} h^{(k-1)}_u) \ + \ W^{(k)}_{negih} \Sigma \ h^{(k)}_\nu \ + \ b^{(k)}$$