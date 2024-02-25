# Node Embedding

#### Encoder Decoder Perspective:
Here, the encoder maps each node in the graph into a low-dimensional node embedding which later used by the decoder to Reconstruction the original graph information.

Encoder is a function that maps nodes to a vector in the vector space of real numbers. Here, the encoder maps the node ID to node embedding. Often this is done using shadow embedding approach. $$ENC(\nu) = Z[\nu]$$
Decoder does the reverse by retaining the original information from the Graphs. This can be given by: $$DEC(ENC(u), ENC(\nu)) = DEC(z_u, z_v) \approx S[u,v]$$
![[Node Embedding Encoder-Decoder.png]]

#### Optimising Encoder - Decoder Model:
The optimisation is required to reduce the *reconstruction loss* over a set of node pairs *D*. In high level, this loss can be given as: $$\Sigma \ l(DEC(z_u, z_\nu), S[u, \nu])$$
The loss function *l* can be cross entropy, MSE or similar functions. 

#### Node embedding methods:
There are various methods for embedding nodes in vector space and theory might differ with each other by decoding function, graph-based similarity measure or loss function. 

1. **Laplacian Eigenmaps:** LE approach is an early technique here L2 distance between two node embedding is calculated. The decoder functions is gives as: $$DEC(z_u, z_v) = ||z_u, z_v||^2_2$$ The final loss function is given as: $$L = \Sigma \ DEC(z_u, z_v) . S[u, \nu]$$
2. **Inner Product methods:** In this approach we consider that similarity between two nodes is directly proportional to the dot products of their embedding. $$DEC(z_u, z_v) = z_u \top z_v$$ The loss function is gives as: $$L = \Sigma \ ||DEC(z_u, z_v) - S[u,v]||^2_2$$ GraRep and HOPE are examples of such algorithms.
   
3. **Random Walk Embedding:** Rather than adopting stochastic rather than deterministic approach.
   - **DeepWalk and node2vec:** Here, the node reconstruction is not done with a polynomial function rather approaches optimise embedding to encode the statistics of random walks. The Decoder for DeepWalk is as follows: $$DEC(z_u, z_v) = \frac{e^{z_u \top z_v}}{\Sigma e^{z_u \top z_v}} \approx p_{G,T}(\nu | u)$$ The loss function is given as: $$L = \Sigma \ -log(DEC(z_u, z_v))$$
#### Limitations of Shadow embedding:
1. The encoder doesn't use all the useful featured of the nodes in the graphs to create the embedding. Due to this potential information is left useless.
2. It can generate embedding for nodes that were present during the training phase and for test data, it is not possible unless additional optimisation is performed. This prevents shallow embedding being used on inductive applications.

# Multi relational data node embedding

#### Knowledge Graph:
Graphs can be used a real word relation representation rather than just being abstract. In that case, the older methods of using node embedding can be used as the edges here represent some information or *knowledge*. They can be represented as $(u, \tau, \nu)$, where $\tau$ represents the knowledge. 
#### Reconstructing Multi Relational Data:
Similar to the case of simple graphs, embedding multi-relational graphs can be viewed as a reconstruction task. The goal is to learn low-dimensional embedding for nodes that can accurately reconstruct the relationships between them. However, in multi-relational graphs, we face the additional challenge of dealing with diverse types of edges or relations.

To tackle this challenge, we enhance the decoder component of our model to make it multi-relational. Instead of solely considering pairs of node embedding, the decoder now takes into account both node embedding and relation types. This enables the model to predict the likelihood of the existence of an edge between two nodes, given their embedding and the relation type. Various decoder functions have been proposed, including RESCAL, TransE, TransX, DistMult, ComplEx, and RotatE. Each decoder offers a unique approach to encoding and decoding relations between nodes.

Choosing an appropriate loss function is crucial for training the model in multi-relational node embedding. One popular and efficient choice is the cross-entropy loss with negative sampling. This loss function compares the predicted probabilities of true and false edges in the graph. By sampling negative examples and employing logistic regression, the model can learn to differentiate between true and false relations more effectively.