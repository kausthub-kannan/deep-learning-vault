# Graphs - Mathematical interpretations

#### Graphs:
Formally,a graph $G=(\nu, \varepsilon)$ is defined by a set of nodes $\nu$ and a set of edges $\varepsilon$ between these nodes.

**Note:**
- Graphs are conveniently represented as Adjacent  Matrices. The node number acts as the indices of the matrix.
- Graphs can be classified as undirected and directed graphs depending on if the edges point towards a specific node or not.
- Graph nodes can have weights, where each nodes have real values assigned to them rather than arbitrary value of {0,1}.

#### Types of Graphs:
1. **Multi Relationship Graphs:** Edges of graphs can represent different relationship between two nodes. Ex: Edges can represent different mode of transportation for each node being a city
2. **Heterogeneous Graphs:** Considering two set of nodes, the graph is called Heterogeneous if intersection of both the set is null. If there are no edges connecting nodes of same set then the heterogeneous graph is termed as **Multipartite Graph**
3. **Multiplex Graphs:** Every node is assumed to belong to every layer, and each layer corresponds to a unique relation, representing the intra-layer edge type for that layer. We also assume that inter-layer edges type scan exist,which connect the same node across layers. Ex: Each layer can represent type of transportation and edges the different paths or connection between the cities.

#### Machine Learning tasks on Graphs:
- Node Classification
- Link Prediction
- Community or Cluster detection
- Graph classification or regression

#### Node Statistics
1. **Node Degree:** Degree of Node is defined as number of edges connected to a node (in case of undirected) or the net number of edges connected to a node (in case of directed). $$d_u = \Sigma A[u, \nu]$$
2. **Node Centrality:** Node centrality defines the importance of the node by taking into account of the recurrence relation with it's nodes.  There are multiple ways of representing node centrality, the below shows approach using *eigen vector centrality.*$$e_u = \frac{1}{\lambda}A[u, \nu]e_\nu$$
   where $e_\nu$ represents the edges connected to the node $\nu$.
4. **Clustering Coefficient:** Nodes sometimes need to be differentiated on the cluster it is located at. Clustering coefficient can be given but the ratio of number of nodes between the neighbouring nodes to that of number of pairs of nodes in neighbourhood. $$c_u = \frac{|(\nu_1, \nu_2 \ \varepsilon: \nu_1, \nu_2 \ \varepsilon \ N(u)|}{\frac {d_u}{2}}$$
#### Graph Statistics (Weisfeiler-Lehman kernel):
Often it is used to compare Graphs and plays a crucial rule in mathematical approach in using Graphs in real world problems. The kernel is calculated in an iterative approach with each iterations following the below steps:
1. Initially, each label is assigned to the nodes. 
2. Here, the labels of the nodes are assigned a set of labels of adjacent nodes.
3. Hashing function is used to compress these set of labels to a new label
4. Feature vector is created by counting the occurrence of each label in original graph and in modified graph.

![[Weisfeiler-Lehman kernel.png]]

#### Neighbourhood Overlap Detection - Relationship Statistics
Node and Graph statistics do not cover the relationship aspect of two nodes which plays a key role in Graph data. An example of relationship statistics is finding out common nodes connected to two set of nodes.

The likely wood find an edge between the node *v* and *u* is directly proportional to the Neighbourhood Overlap Detection.

1. **Local Overlap Measures:**
   It is the simple definition of finding common neighbour between two nodes. Often there are several ways to measure this:
   - **Sorenson Index:** $\frac{2|N(u) \bigcap N(v)|}{d_u + d_v}$ 
   - **Salton Index:** $\frac{2|N(u) \bigcap N(v)|}{\sqrt{d_u d_v}}$
   - **Jaccard Index:** $\frac{2|N(u) \bigcap N(v)|}{2|N(u) \bigcup N(v)|}$
   - **Resource Allocation Index:** $\Sigma \frac{1}{d_u}$
   - **Adamic-Adar (AA):** $\Sigma \frac{1}{log(d_u)}$
   
   The issue with Local overlap is that it cannot consider the holistic view of the graph. Example, two nodes might not have local overlap but still belong to same cluster.

2. **Global Overlap Measures:**
   Global measure solves the issue of local measure where instead of neighbouring nodes we consider paths.
   - **Katx Index:** $\Sigma \beta ^ i \ A^i \ [u,v]$; where $\beta$ is user defined weight. In terms of vectors, this can be given as $(I - \beta A)^{-1} - I$
   - **LHN Similarity:** Katx index has higher bias with node's with higher degree. To solve this LHN takes up the ratio between actual path between the nodes and expected path between the nodes which is given by:  $\frac{A^i}{E[A^i]}$ where $E[A^i]=\frac{d_u d_v}{2m}$ and *m* is total number of edges. The simplified form of LHN is given by: $S_{LHN}=2\alpha m\lambda {-1}D^{-1} (I - \frac{\beta}{\lambda_1}A)^{-1} D^{-1}$