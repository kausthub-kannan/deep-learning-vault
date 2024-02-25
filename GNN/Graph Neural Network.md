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
1. Self Loops and Normalization (GCN): $h^{(k)}_u = \sigma (W^{(k)}_{self} \Sigma \frac{h^{(k-1)}_u)}{\sqrt{|N(u)||N(\nu)|}})$
2. Aggregation via MLP: $m_{N(u)} = MLP_\theta(\Sigma MLP_\phi (h_\nu))$
3. Aggregation via Attention: $m_{N(u)} = \Sigma \alpha_{u,\nu}h_\nu$
4. RNN Approach: $h^{(k)}_u = GRU(h^(k-1)_u)$

## Self Loops
Self loops can be used to eliminate the explicit update step.  This can simplify the equation to below: $$h^{(k)}_u = AGGREGATE({h^{(k-1)}_\nu, \forall \nu \in N(u) \ \cup \ {u}})$$
Here, the aggregation takes place such that the update takes place directly to the node including node's neighbors.

## Neighborhood Normalization
To keep GNN more practically possible, we require to add the normalization step. This is easily doable via applying Neighborhood Normalization. For $u$ nodes. just applying general normalization works but a better way to do it would be applying symmetric normalization which is given as: $$m_{N(u)} = \Sigma \frac{h_\nu}{\sqrt{|N(u)||N(\nu)|}}$$
Combining both, we get GCNs - Graph Convolutional Networks $$h^{(k)}_u = \sigma (W^{(k)}_{self} \Sigma \frac{h^{(k-1)}_u)}{\sqrt{|N(u)||N(\nu)|}})$$
## Set Aggregators
To improvise aggregation, three approaches are often used:
1. Set Pooling: $m_{N(u)} = MLP_\theta (\ \Sigma \ MLP_\phi (h_\nu) \ )$
2. Janossy Pooling: $m_{N(u)} = MLP_\theta (\frac{1}{|\Pi|} \Sigma p_\phi (h_\nu))$
3. GAT (Graph Attention Networks): $m_{N(u)} = \Sigma \ \alpha_{u,\nu}, h_\nu$ where  $[ \alpha_{u}, v = \frac{\exp(a^T[W_h^u \oplus W_h^v])}{\sum_{v' \in N(u)} \exp(a^T[W_h^u \oplus W_h^v])} ]$

## Skip Connections
GNNs face over-smoothing issue. Over-smoothing occurs when node-specific information is lost or "washed-out" after several iterations. *Skip Connections* are used to solve this issue which directly preserve previous round information. This is done with a very simple approach where we just concatenate the previous node information with new node information. Along with the concatenation we can apply gate vectors similar to linear interpolation. $$UPDATE \ interpolate(hu,mN(u)) = \alpha _ {1} \cdot UPDATE_ {base} (h_ {u} , m_ {N} (u)) + \alpha _ {2} \odot h$$
## Graph Pooling

![[Graph pooling.png]]

Pooling plays important role in convolutional networks which is useful to learn entire embedding of a graph rather than just a node or set of nodes. 