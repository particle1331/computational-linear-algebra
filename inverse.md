## Matrix inverse and pseudoinverse

<br>

* (7.1) **Full rank iff. invertible.** Let $\mathbf A \in \mathbb R^{n \times n}.$ TFAE
  1. $\text{rank }\mathbf A = n.$
  2. $\mathbf A$ is one-to-one.
  3. $\mathbf A$ is onto.
  4. $\mathbf A$ is invertible.
  5. $\mathbf A$ is nonsingular.

  This can be proved using the rank-nullity theorem which constrains rank to be $n$ if and only if the dimension of the null space is zero. The latter is then equivalent to $\mathbf A$ being one-to-one, while the former to $\mathbf A$ being onto. This proves (1-4). A matrix is nonsingular if it has no nonzero singular value so that the image of the unit sphere under $\mathbf A$ is non-degenerate. Note that $\mathbf A$ is invertible if and only if $\mathbf \Sigma$ is invertible (since $\mathbf U$ and $\mathbf V$ are invertible). But $\mathbf \Sigma$ is invertible if and only if $r = n,$ and we can take ${\mathbf\Sigma^{-1}}_ {ii} = \sigma_i^{-1}$ for $i = 1, \ldots, n.$ Thus, a matrix is invertible if and only if it is nonsingular. This proves (5) $\iff$ (4).

<br>

* (7.2) **Sparse matrix has a dense inverse.** A sparse matrix can have a dense inverse. This can cause memory errors in practice. In `src/9_sparse_to_dense.py` we artificially construct a sparse matrix. This is typically singular, so we shift it to make it nonsingular.  The result is that the inverse is 50x more dense than the original matrix:

  ```
  A sparsity:      0.001999
  A_inv sparsity:  0.108897
  ```

<br>

* (7.3) **Existence of left and right inverses.** Let $\mathbf A \in \mathbb R^{m\times n}.$ TFAE 
  1. $\text{rank }\mathbf A = n.$
  2. $\mathbf A$ is 1-1.
  3. $\mathbf A$ is left invertible.
  4. $\mathbf A$ has $n$ nonzero singular values.

  It's easy to see that (1-3) are equivalent and (1) $\implies$  (4). We prove (3) $\impliedby$ (4). Suppose $\mathbf A$ has $n$ nonzero singular values, then we can construct a left inverse using the SVD $\mathbf A = \mathbf U \mathbf \Sigma \mathbf V^\top$ which allows us to write
  $
  \mathbf A^\top \mathbf A = \mathbf V \mathbf \Sigma^\top \mathbf \Sigma \mathbf V^\top.
  $ 
  This is invertible since $r = n,$ i.e. $\mathbf\Sigma^\top \mathbf \Sigma$ is $n \times n$ with nonzero entries on its diagonal. Moreover, the inverse can be efficiently computed using $(\mathbf A^\top \mathbf A)^{-1} = \mathbf V (\mathbf\Sigma^\top \mathbf \Sigma)^{-1} \mathbf V^\top$ where $(\mathbf\Sigma^\top \mathbf \Sigma)^{-1} = \mathbf \Sigma_n^{-2}$ is the diagonal matrix with entries $\sigma_j^{-2}$ for $j=1,\ldots, n.$ Then, a left inverse for $\mathbf A$ is
    $$
    (\mathbf A^\top \mathbf A)^{-1} \mathbf A^\top.
    $$ 

  We have corresponding dual equivalences about the rows of $\mathbf A.$ In this case, $\mathbf A$ is onto, and we have a wide matrix with maximal rows. A right inverse of $\mathbf A$ can be constructed as
  $$
  \mathbf A^\top (\mathbf A \mathbf A^\top)^{-1}
  $$
  where $(\mathbf A \mathbf A^\top)^{-1} = \mathbf U (\mathbf \Sigma \mathbf \Sigma^\top)^{-1} \mathbf U^\top$ can be efficiently computed as in the left inverse with $(\mathbf \Sigma \mathbf \Sigma^\top)^{-1} = \mathbf \Sigma_m^{-2}.$

<br>

* (7.4) **Moore-Penrose Pseudo-inverse.** Now that we know how to compute the one sided inverse from rectangular matrices, assuming they have full column rank or full row rank, the big missing piece is what to do with a reduced rank matrix. It turns out that it is possible to find another matrix that is not formally an inverse, but is some kind of a good approximation of what the inverse element should be in a least squares sense (later), i.e. what is called a pseudo-inverse. The **Moore-Penrose pseudo-inverse** for a matrix $\mathbf A \in \mathbb R^{m \times n}$ is defined as the unique matrix $\mathbf A^+ \in \mathbb R^{n \times m}$ that satisfies the four Penrose equations:

  1. $\mathbf A \mathbf A^+ \mathbf A = \mathbf A$
  2. $\mathbf A^+ \mathbf A \mathbf A^+ = \mathbf A^+$
  3. $\mathbf A \mathbf A^+$ is symmetric.
  4. $\mathbf A^+ \mathbf A$ is symmetric.

  These properties make $\mathbf A^+$ look like an inverse of $\mathbf A$. In fact, if $\mathbf A$ is invertible, then $\mathbf A^{-1}$ trivially satisfies the equations (also see below for left and right inverses). The Moore-Penrose pseudo-inverse exists (from the SVD below) and is [unique](https://en.wikipedia.org/wiki/Proofs_involving_the_Moore%E2%80%93Penrose_inverse) for every rectangular matrix even rank deficient ones.
  <br><br>
  **Existence.** Consider the SVD $\mathbf A = \mathbf U \mathbf \Sigma \mathbf V^\top,$ we naturally take
  $$
    \mathbf A^{+} = \mathbf V \mathbf \Sigma^+ \mathbf U^\top
  $$
  where $\mathbf \Sigma^+$ is the unique matrix that satisfies the Penrose equations for $\mathbf \Sigma.$ This turns out to be  the diagonal matrix of shape $n \times m$ that is a block matrix with the upper left block $\mathbf \Sigma_r^{-1}$ and zero blocks elsewhere. That is, $\mathbf\Sigma^+\mathbf \Sigma$ and  $\mathbf\Sigma\mathbf \Sigma^+$ with $\mathbf I_r$ on the upper left block and zero blocks elsewhere are symmetric, then $\mathbf \Sigma \mathbf\Sigma^+\mathbf \Sigma = \mathbf \Sigma$ and $\mathbf \Sigma^+ \mathbf \Sigma \mathbf \Sigma^+ = \mathbf \Sigma^+.$ It follows that $\mathbf A^+$ is the Moore-Penrose pseudo-inverse for $\mathbf A,$ e.g.  $\mathbf A \mathbf A^+ = \mathbf U_r\; {\mathbf U_r}^\top$ and $\mathbf A^+ \mathbf A = \mathbf V_r\; {\mathbf V_r}^\top$ are symmetric, and the first two Penrose equations follows from the same two equations for $\mathbf \Sigma^+.$ This is precisely how `np.linalg.pinv` calculates the pseudo-inverse $\mathbf A^+$:
  <br>
    ```python
    In [1]: import numpy as np
    
    In [2]: A = np.random.randn(3, 3)
    In [3]: A[:, 0] = A[:, 1] * 3.2 - A[:, 2] * 1.2 # make rank 2
    In [4]: u, s, vT = np.linalg.svd(A)                                     
    In [5]: s_pinv = np.diag([ 1/x if x > 1e-8 else 0 for x in s ])
    In [6]: vT.T @ s_pinv @ u.T
    Out[6]: 
    array([[-0.13374723, -0.01096947,  0.08071149],
           [ 0.0228385 , -0.16845822, -0.12487767],
           [ 0.1723587 , -0.44008068, -0.40026669]])

    In [7]: np.linalg.pinv(A)
    Out[7]: 
    array([[-0.13374723, -0.01096947,  0.08071149],
           [ 0.0228385 , -0.16845822, -0.12487767],
           [ 0.1723587 , -0.44008068, -0.40026669]])
    ```

<br>

<p align="center">
      <img src="img/9_sigma_pseudoinverse.png" width=80%/> <br>
      <b>Figure.</b> Pseudo-inverse of the singular values matrix.
</p> 

<br>

* (7.5) **Moore-Penrose pseudo-inverse as left and right inverse.** Let $\mathbf A \in \mathbb R^{m \times n}$ with maximal rank. It turns out the left and right inverses we constructed above is the Moore-Penrose pseudo-inverse of $\mathbf A$ in each case:

  * $\mathbf A^+ = (\mathbf A^\top \mathbf A)^{-1} \mathbf A^\top$ (tall)
  
  * $\mathbf A^+ = \mathbf A^\top(\mathbf A \mathbf A^\top)^{-1}$ (wide) 

  This follows from uniqueness and the fact that the left and right inverses each satisfies the Penrose equations. Any left or right inverse will trivially satisfy the first two equations, but not both the third and fourth! For example:
  <br>
  ```python
  In [31]: A = np.vstack([ np.eye(3), [0, 0, 1] ])
  In [33]: B = np.hstack([ np.eye(3), [[0], [0], [0]] ])     
  In [34]: A @ B       
  Out[34]: 
  array([[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 1., 0.]])
  In [35]: B @ A        
  Out[35]: 
  array([[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]])
  ``` 

<br>
  
* (7.6) **An exercise on consistency.** Recall that $\mathbf A^+ = \mathbf V \mathbf \Sigma^+ \mathbf U^\top$ uniquely. As an exercise, we want to show that this is consistent with the formula $\mathbf A^+ = (\mathbf A^\top \mathbf A)^{-1} \mathbf A^\top$ which is true for matrices with linearly independent columns. We do this for the tall case $m > n$, the case where the matrix is wide is analogous. Then 
    $$
    \mathbf A^+ = (\mathbf A^\top \mathbf A)^{-1} \mathbf A^\top
    = \mathbf V (\mathbf \Sigma^\top \mathbf \Sigma)^{-1} \mathbf \Sigma^\top \mathbf U^\top.
    $$
    Since $\mathbf \Sigma$ is a tall matrix having linearly independent columns, we have $\mathbf \Sigma^+ = (\mathbf \Sigma^\top \mathbf \Sigma)^{-1} \mathbf \Sigma^\top.$ Thus, $(\mathbf A^\top \mathbf A)^{-1} \mathbf A^\top = \mathbf V \mathbf \Sigma^+\mathbf U^\top.$ We get the same agreement when $\mathbf A$ is right invertible. This completes the exercise.

<br>
