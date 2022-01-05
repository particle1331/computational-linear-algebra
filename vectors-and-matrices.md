## Vectors and matrices

<br>

  - (1.1) **No. of linearly independent vectors in ${\mathbb R^m}$**.  The maximum length $n$ of a list of linearly independent vectors in $\mathbb R^m$ is bounded by $m$.  If $n > m$, then the list is linearly dependent. 
  
<br>
  
  - (1.2) **Complexity of checking independence**.
  Suppose $n \leq m.$ What is the time complexity of showing n vectors in $\mathbb R^m$ are linearly independent? i.e. solving for nonzero solutions to $\mathbf A\mathbf x = \mathbf 0$. For instance, we have $\mathcal{O}(\frac{2}{3} mn^2)$ using Gaussian elimination assuming $\mathcal{O}(1)$ arithmetic which is a naive assumption as careless implementation can easily create numbers that can be [exponentially large](https://cstheory.stackexchange.com/questions/3921/what-is-the-actual-time-complexity-of-gaussian-elimination)! In practice, the best way to compute the rank of $\mathbf A$ is through its SVD. This is, for example, how `numpy.linalg.matrix_rank` is implemented.

<br>

  - (1.3) **Basis is non-unique.**
  A choice of basis is non-unique but gives unique coordinates for each vector once the choice of basis is fixed. Some basis are better than others for a particular task, e.g. describing a dataset better. There are algorithms such as PCA & ICA that try to minimize some objective function.
  
<br>

  * (1.4) **Orthogonal matrices are precisely the linear isometries of $\mathbb R^n$.** A matrix $\mathbf A \in \mathbb R^n$ is an isometry if $\lVert \mathbf A \mathbf x\rVert^2 = \lVert \mathbf x \rVert^2$ for all $\mathbf x \in \mathbb R^n$. Note that $\lVert \mathbf A \mathbf x \rVert^2 = \mathbf x^\top\mathbf A^\top \mathbf A \mathbf x$ and $\lVert \mathbf x \rVert^2 = \mathbf x^\top \mathbf x$. So orthogonal matrices are isometries. Conversely, if a matrix $\mathbf A$ is an isometry, we can let $\mathbf x = \mathbf e_i - \mathbf e_j$ to get $\mathbf e_i^\top (\mathbf A^\top \mathbf A) \mathbf e_j = (\mathbf A^\top \mathbf A)_ {ij} = \delta_ {ij}$ where $\delta_{ij}$ is the Kronecker delta or $\mathbf A^\top\mathbf A = \mathbf I$. This tells us that length preserving matrices in $\mathbb R^n$ are necessarily orthogonal. Orthogonal matrices in $\mathbb R^2$ are either rotations or reflections &mdash; both of these are length preserving. The more surprising result is that these are the only length preserving matrices in $\mathbb R^2$!

<br>

  - (1.5) **Orthogonal matrices as projections.** An orthogonal matrix $\mathbf U$ is defined as a matrix with orthonormal vectors in its column. It follows $\mathbf U^\top \mathbf U = \mathbf I.$ Since $\mathbf U$ is invertible, we can use uniqueness of inverse to get $\mathbf U \mathbf U^\top = \mathbf I.$ However, we can obtain this latter identity geometrically. Let $\mathbf x$ be a vector, then $\mathbf x = \sum_i \mathbf u_i \mathbf u_i^\top \mathbf x.$ This is true by uniqueness of components in a basis. Thus, $\mathbf U \mathbf U^\top \mathbf x = \mathbf x$ for any $\mathbf x,$ or $\mathbf U \mathbf U^\top  = \mathbf I.$ 
  
<br>

  - (1.6) **Shifting a matrix away from degeneracy:**
  $\mathbf{A} + \lambda \mathbf{I} \simeq \mathbf{A}.$ 
  Geometric interpretation: inflate a matrix from a degenerate plane towards being a sphere. This is a form of regularization. 

<br>

  - (1.7) **Calculating the Hermitian transpose in Python.** Let `A` be a numpy array. The following calculates the Hermitian transpose:
    1. `np.conj(A).T`
    2. `np.conj(A.T)`

<br>

* (1.8) Four ways of matrix multiplication: 

  - **Outer product perspective** <br> 
    ```
    AB[i, j] = sum(k, A[i, k] B[k, j]) 
             = sum(k, outer(A[:, k], B[k, :])[i, j]
    ```
  - **Row perspective**: <br> 
    ```
    AB[i, :] = sum(k, A[i, k] B[k, :]) 
    ```
  - **Column perspective**: <br> 
    ```
    AB[:, j] = sum(k, A[:, k] B[k, j]) 
    ```
