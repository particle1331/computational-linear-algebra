## Matrix inverse and pseudoinverse

<br>

* (7.1) **Full rank iff. invertible.** Let $\bold A \in \mathbb R^{n \times n}.$ TFAE
  1. $\text{rank }\bold A = n.$
  2. $\bold A$ is one-to-one.
  3. $\bold A$ is onto.
  4. $\bold A$ is invertible.
  5. $\bold A$ is nonsingular.

  This can be proved using the rank-nullity theorem which constrains rank to be $n$ if and only if the dimension of the null space is zero. The latter is then equivalent to $\bold A$ being one-to-one, while the former to $\bold A$ being onto. This proves (1-4). A matrix is nonsingular if it has no nonzero singular value so that the image of the unit sphere under $\bold A$ is non-degenerate. Note that $\bold A$ is invertible if and only if $\bold \Sigma$ is invertible (since $\bold U$ and $\bold V$ are invertible). But $\bold \Sigma$ is invertible if and only if $r = n,$ and we can take ${\bold\Sigma^{-1}}_ {ii} = \sigma_i^{-1}$ for $i = 1, \ldots, n.$ Thus, a matrix is invertible if and only if it is nonsingular. This proves (5) $\iff$ (4).

<br>

* (7.2) **Sparse matrix has a dense inverse.** A sparse matrix can have a dense inverse. This can cause memory errors in practice. In `src/9_sparse_to_dense.py` we artificially construct a sparse matrix. This is typically singular, so we shift it to make it nonsingular.  The result is that the inverse is 50x more dense than the original matrix:

  ```
  A sparsity:      0.001999
  A_inv sparsity:  0.108897
  ```

<br>

* (7.3) **Existence of left and right inverses.** Let $\bold A \in \mathbb R^{m\times n}.$ TFAE 
  1. $\text{rank }\bold A = n.$
  2. $\bold A$ is 1-1.
  3. $\bold A$ is left invertible.
  4. $\bold A$ has $n$ nonzero singular values.

  It's easy to see that (1-3) are equivalent and (1) $\implies$  (4). We prove (3) $\impliedby$ (4). Suppose $\bold A$ has $n$ nonzero singular values, then we can construct a left inverse using the SVD $\bold A = \bold U \bold \Sigma \bold V^\top$ which allows us to write
  $
  \bold A^\top \bold A = \bold V \bold \Sigma^\top \bold \Sigma \bold V^\top.
  $ 
  This is invertible since $r = n,$ i.e. $\bold\Sigma^\top \bold \Sigma$ is $n \times n$ with nonzero entries on its diagonal. Moreover, the inverse can be efficiently computed using $(\bold A^\top \bold A)^{-1} = \bold V (\bold\Sigma^\top \bold \Sigma)^{-1} \bold V^\top$ where $(\bold\Sigma^\top \bold \Sigma)^{-1} = \bold \Sigma_n^{-2}$ is the diagonal matrix with entries $\sigma_j^{-2}$ for $j=1,\ldots, n.$ Then, a left inverse for $\bold A$ is
    $$
    (\bold A^\top \bold A)^{-1} \bold A^\top.
    $$ 

  We have corresponding dual equivalences about the rows of $\bold A.$ In this case, $\bold A$ is onto, and we have a wide matrix with maximal rows. A right inverse of $\bold A$ can be constructed as
  $$
  \bold A^\top (\bold A \bold A^\top)^{-1}
  $$
  where $(\bold A \bold A^\top)^{-1} = \bold U (\bold \Sigma \bold \Sigma^\top)^{-1} \bold U^\top$ can be efficiently computed as in the left inverse with $(\bold \Sigma \bold \Sigma^\top)^{-1} = \bold \Sigma_m^{-2}.$

<br>

* (7.4) **Moore-Penrose Pseudo-inverse.** Now that we know how to compute the one sided inverse from rectangular matrices, assuming they have full column rank or full row rank, the big missing piece is what to do with a reduced rank matrix. It turns out that it is possible to find another matrix that is not formally an inverse, but is some kind of a good approximation of what the inverse element should be in a least squares sense (later), i.e. what is called a pseudo-inverse. The **Moore-Penrose pseudo-inverse** for a matrix $\bold A \in \mathbb R^{m \times n}$ is defined as the unique matrix $\bold A^+ \in \mathbb R^{n \times m}$ that satisfies the four Penrose equations:

  1. $\bold A \bold A^+ \bold A = \bold A$
  2. $\bold A^+ \bold A \bold A^+ = \bold A^+$
  3. $\bold A \bold A^+$ is symmetric.
  4. $\bold A^+ \bold A$ is symmetric.

  These properties make $\bold A^+$ look like an inverse of $\bold A$. In fact, if $\bold A$ is invertible, then $\bold A^{-1}$ trivially satisfies the equations (also see below for left and right inverses). The Moore-Penrose pseudo-inverse exists (from the SVD below) and is [unique](https://en.wikipedia.org/wiki/Proofs_involving_the_Moore%E2%80%93Penrose_inverse) for every rectangular matrix even rank deficient ones.
  <br><br>
  **Existence.** Consider the SVD $\bold A = \bold U \bold \Sigma \bold V^\top,$ we naturally take
  $$
    \bold A^{+} = \bold V \bold \Sigma^+ \bold U^\top
  $$
  where $\bold \Sigma^+$ is the unique matrix that satisfies the Penrose equations for $\bold \Sigma.$ This turns out to be  the diagonal matrix of shape $n \times m$ that is a block matrix with the upper left block $\bold \Sigma_r^{-1}$ and zero blocks elsewhere. That is, $\bold\Sigma^+\bold \Sigma$ and  $\bold\Sigma\bold \Sigma^+$ with $\bold I_r$ on the upper left block and zero blocks elsewhere are symmetric, then $\bold \Sigma \bold\Sigma^+\bold \Sigma = \bold \Sigma$ and $\bold \Sigma^+ \bold \Sigma \bold \Sigma^+ = \bold \Sigma^+.$ It follows that $\bold A^+$ is the Moore-Penrose pseudo-inverse for $\bold A,$ e.g.  $\bold A \bold A^+ = \bold U_r\; {\bold U_r}^\top$ and $\bold A^+ \bold A = \bold V_r\; {\bold V_r}^\top$ are symmetric, and the first two Penrose equations follows from the same two equations for $\bold \Sigma^+.$ This is precisely how `np.linalg.pinv` calculates the pseudo-inverse $\bold A^+$:
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

* (7.5) **Moore-Penrose pseudo-inverse as left and right inverse.** Let $\bold A \in \mathbb R^{m \times n}$ with maximal rank. It turns out the left and right inverses we constructed above is the Moore-Penrose pseudo-inverse of $\bold A$ in each case:

  * $\bold A^+ = (\bold A^\top \bold A)^{-1} \bold A^\top$ (tall)
  
  * $\bold A^+ = \bold A^\top(\bold A \bold A^\top)^{-1}$ (wide) 

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
  
* (7.6) **An exercise on consistency.** Recall that $\bold A^+ = \bold V \bold \Sigma^+ \bold U^\top$ uniquely. As an exercise, we want to show that this is consistent with the formula $\bold A^+ = (\bold A^\top \bold A)^{-1} \bold A^\top$ which is true for matrices with linearly independent columns. We do this for the tall case $m > n$, the case where the matrix is wide is analogous. Then 
    $$
    \bold A^+ = (\bold A^\top \bold A)^{-1} \bold A^\top
    = \bold V (\bold \Sigma^\top \bold \Sigma)^{-1} \bold \Sigma^\top \bold U^\top.
    $$
    Since $\bold \Sigma$ is a tall matrix having linearly independent columns, we have $\bold \Sigma^+ = (\bold \Sigma^\top \bold \Sigma)^{-1} \bold \Sigma^\top.$ Thus, $(\bold A^\top \bold A)^{-1} \bold A^\top = \bold V \bold \Sigma^+\bold U^\top.$ We get the same agreement when $\bold A$ is right invertible. This completes the exercise.

<br>
