# Notes
  - (2.30) **No. of linearly independent vectors in ${\mathbb R^m}$**.  The maximum length $n$ of a list of linearly independent vectors in $\mathbb R^m$ is bounded by $m$.  If $n > m$, then the list is linearly dependent. <br><br>
  
  - (2.30) **Complexity of checking independence**.
  Suppose $n \leq m.$ What is the time complexity of showing n vectors in $\mathbb R^m$ are linearly independent? i.e. solving for nonzero solutions to $\bold A\bold x = \bold 0$. For instance, we have $\mathcal{O}(\frac{2}{3} mn^2)$ using Gaussian elimination assuming $\mathcal{O}(1)$ arithmetic which is a naive assumption as careless implementation can easily create numbers with with [exponentially many bits](https://cstheory.stackexchange.com/questions/3921/what-is-the-actual-time-complexity-of-gaussian-elimination)! In practice, the best way to compute the rank of $\bold A$ is through its SVD. This is, for example, how `numpy.linalg.matrix_rank` is implemented.
  <br><br>

  - (2.30) **Basis are non-unique**.
  Non-unique, but gives unique coordinate for each vector when the choice of basis is fixed. Some basis are better than others for a particular task, e.g. describing a dataset better. There are algorithms such as PCA & ICA that try to minimize some objective function.<br><br>

  - (3.34) **Shifting a matrix away from degeneracy**:
  $\bold A + \lambda \bold I = \bold C.$ 
  Geometric interpretation: inflate a matrix from a degenerate plane towards being a sphere. This is a form of regularization.  See (4.51) which shows how linear maps transform the unit circle to an ellipse. A singular matrix $\bold A$ maps the unit circle to a degenerate (flat) ellipse.
  <br><br>

  - (3.35) In the video $\sigma \bold A = \bold A \sigma$. Multiplying a matrix with a scalar $\sigma$ can be interpreted as multiplying with $\sigma \bold I$ where $\bold I$ is the identity matrix of the appropriate size. <br><br>

  - (3.37) **Calculating the Hermitian transpose in Python.** Let `A` be a numpy array. The following calculates the Hermitian transpose:
    1. `np.conj(A).T`
    2. `np.conj(A.T)`
    3. `np.matrix(A).H` (deprecated soon)<br><br>

* (4.44) Cool way of writing the four ways of matrix multiplication: 

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
  <br>

* (4.46) **Eigendecomposition of a matrix**
  $\bold A \bold U = \bold U \bold \Lambda$.
  The eigendecomposition of a matrix is an example of the fact that post-multipyling with a diagonal matrix weights the column &mdash; in this case, the matrix $\bold U$ of eigenvectors of $\bold A$ with the eigenvalues in the diagonal of $\bold \Lambda$.
  <br><br>


* (4.51) **Geometry of linear transformations.** In the code challenge, we saw that a unit circle is mapped into an ellipse. An explanation for this begins by considering the polar decomposition of any real matrix $\bold A$ into $\bold A =  \bold Q \bold P$ where $\bold Q \in \mathbb{R}^{n\times n}$ is orthogonal and $\bold P \in \mathbb{R}^{n\times n}$ is symmetric positive semidefinite. Note that $\bold P$ is a self-adjoint operator on $\mathbb R^n$. So $\bold P$ has real eigenvalues [[Prop. 7.6]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf) that are nonnegative since $\bold P$ is semidefinite. Self-adjointness further implies that $\bold P$ has $n$ real eigenvalues and $n$ real orthonormal eigenvectors [[Thm. 8.3]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf). This is called the **real spectral theorem** which allows us to write $\bold P = \bold V \bold \Sigma \bold V^\top$ where $\bold V$ is the orthogonal matrix formed by column stacking its eigenvectors of $\bold P$, and $\bold \Sigma$ is the diagonal matrix of its eigenvalues. If we look at $\bold V$ and $\bold V^\top$ as change of basis matrices, or writing $\bold P \bold x = \sum_{i=1}^n \lambda_i \bold v_i \bold v_i^\top \bold x$ (projection operators!) we see that $\bold A$ stretches $\mathbb R^n$ in the direction of the $n$ orthogonal eigenvectors, followed by an orthogonal transformation by $\bold Q$. We can [calculate](https://math.stackexchange.com/a/2924263) that orthogonal transformations of $\mathbb R^2$ are either rotations or reflections, so the unit circle in $\mathbb R^2$ is stretched onto an ellipse that is either reflected or rotated resulting to the final ellipse! Note that the rank of $\bold A$ is equal to the rank of $\bold P$ since $\bold Q$ is nonsingular. So whenever $\bold A$ doesn't have full rank, some eigenvalues of $\bold P$ are zero, which results to a degenerate ellipse, i.e. an ellipse collapsing on some axis (see below).
<br><br>
**Remark.** The polar decomposition comes from the SVD which states that any real matrix $\bold A \in \mathbb R^{m \times n}$ can be written as $\bold A = \bold U \bold \Sigma \bold V^\top$ where $\bold U \in \mathbb R^{m \times m}$ and $\bold V \in \mathbb R^{n \times n}$ are real orthonogonal matrices and $\bold\Sigma  \in \mathbb R^{m \times n}$ is a diagonal matrix with nonnegative real numbers on the diagonal are the **singular values** of $\bold A$. The number of nonzero singular values is equal to the rank of $\bold A$. Observe that writing $\bold A = \bold Q \bold P$ where $\bold Q = \bold U \bold V^\top$ and $\bold P = \bold V\bold \Sigma \bold V^\top$ gives us the polar decomposition. <br><br>

* (4.51) **Computing the polar decomposition.** In `src\4_polar_decomp.py`, we verify the theory by calculating the polar decomposition: `Q, P = scipy.linalg.polar(A)`. The result is that `P` is real symmetric and `Q` is real orthogonal. Some eigenvalues of `P` are zero for singular `A`  (right) while all are nonzero for nonsingular `A` (left). The eigenvectors of `P` are scaled by the corresponding eigenvalues, then rotated with `Q`. The rotated eigenvectors of `P` lie along the major and minor axis of the ellipse: the directions where the circle is stretched prior to rotation. This checks out! The code also includes a calculation of the polar decomposition from the SVD. <br><br>
    <p align="center">
    <img src="img/4_polar_decomposition.png" title="drawing" width="600" />
    </p> <br>
    
    **Remarks.**
    This gives us the intuition that $\bold P$ in the polar decomposition must be unique. Indeed, $\bold P^2 = \bold A^\top \bold A$. Since $\bold A^\top \bold A$ is evidently symmetric  positive semidefinite, it has a unique symmetric positive semidefinite square root $\sqrt{\bold A^\top \bold A}$ [[Thm. 3]](https://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect7.pdf). Then, $\bold P = \sqrt{\bold A ^\top \bold A}$ by uniqueness. It follows that the polar decomposition can be written as $\bold A = \bold U \sqrt{\bold A^\top \bold A}$ for some orthogonal matrix $\bold U$ &mdash; cf. [[Lem. 9.6]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf) which states the polar decomposition in terms of the existence such an isometry.
    The matrix $\bold U$ is only unique if $\bold A$ is nonsingular. For instance, if $\bold A$ is singular, then we can reflect across the axis where the space is collapsed and still get the same transformation.<br><br> 

* **Orthogonal matrices in $\mathbb R^n$ are precisely the linear isometries of $\mathbb R^n$.** A matrix $\bold A \in \mathbb R^n$ is an isometry if $\lVert \bold A \bold x\rVert^2 = \lVert \bold x \rVert^2$ for all $\bold x \in \mathbb R^n$. Note that $\lVert \bold A \bold x \rVert^2 = \bold x^\top\bold A^\top \bold A \bold x$ and $\lVert \bold x \rVert^2 = \bold x^\top \bold x$. So orthogonal matrices are isometries. Conversely, if a matrix $\bold A$ is an isometry, we can let $\bold x = \bold e_i - \bold e_j$ to get $\bold e_i^\top (\bold A^\top \bold A) \bold e_j = 
(\bold A^\top \bold A)_{ij} 
= \delta_{ij}$ where $\delta_{ij}$ is the Kronecker delta or $\bold A^\top\bold A = \bold I$. This tells us that length preserving matrices in $\mathbb R^n$ are necessarily orthogonal. Orthogonal matrices in $\mathbb R^2$ are either rotations or reflections &mdash; both of these are length preserving. The more surprising result is that these are the only length preserving matrices in $\mathbb R^2$!
<br><br> 

* **Polar in polar decomposition.** The name "polar decomposition" comes from writing complex numbers as $z = re^{i\theta}$ in terms of its polar coordinates. Here $r = \sqrt{\bar z z }$ (analogous to $\sqrt{\bold A^* \bold A}$) and multiplication by $e^{i\theta}$ is an isometry of $\mathbb C$ (analogous to the isometric property of $\bold U$).<br><br>

* **A nondiagonalizable matrix.** The matrix $\bold A = [[1, 0], [1, 1]]$ has eigenvalues are $\lambda_1 = \lambda_2 = 1$ with eigenvectors of the form $\bold v = [0, t]^\top$ for nonzero $t \in \mathbb R$. It follows that $\bold A$ is not diagonalizable since it has at most one linearly independent eigenvectors &mdash; not enough to span $\mathbb R^2.$ <br><br>

* (4.56) **Symmetric product of two symmetric matrices.** Suppose $\bold S$ and $\bold T$ are symmetric matrices. What is the condition so that their product $\bold S \bold T$ is symmetric? Observe that $(\bold S \bold T)^\top = \bold T ^\top \bold S ^\top = \bold T \bold S.$ Thus, the product of two symmetric matrices is symmetric if and only if the matrices commute. This works for a very small class of matrices, such as zeros or constant diagonal matrices. 
In the case of $2 \times 2$ matrices, this is satisfied most naturally by  matrices with constant diagonal entries &mdash; this is just a quirk that does not generalize to higher dimensional matrices.
The lack of symmetry, i.e. $\bold S \bold T \neq \bold T \bold S$, turns out to be extremely important in machine-learning, multivariate statistics, and signal processing, and is a core part of the reason why linear classifiers are so successful [[Lec 56, Q&A]](https://www.udemy.com/course/linear-algebra-theory-and-implementation/learn/lecture/10738628#questions/13889570/): 
    >  "The lack of symmetry means that $\bold C=\bold B^{-1} \bold A$ is not symmetric, which means that $\bold C$ has non-orthogonal eigenvectors. In stats/data science/ML, most linear classifiers work by using generalized eigendecomposition on two data covariance matrices $\bold B$ and $\bold A$, and the lack of symmetry in $\bold C$ turns a compression problem into a separation problem. I talk about this briefly in lecture 153, which you can skip forwards to if you're curious. But if you are not familiar with eigendecomposition, then it might not make sense until you get to that point linearly."

<br>

* (4.57) Hadamard and standard multiplications are equivalent for diagonal matrices. This can have consequences in practice. The following code in IPython shows that Hadamard multiplication is 3 times faster than standard multiplication in NumPy.
    ```python
    In [1]: import numpy as np
    In [2]: D = np.diag([1, 2, 3, 4, 5])
    In [3]: %timeit D @ D
    2.21 µs ± 369 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    In [4]: %timeit D * D
    717 ns ± 47.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    ```     
<br>

* (4.59) **Frobenius norm.** Let $\bold A$ and $\bold B$ be $m \times n$ matrices. The **Frobenius dot product** between two matrices $\bold A$ and $\bold B$ is defined as 
  $$
  {\langle \bold A, \bold B \rangle} =  {\sum \sum (\bold A \odot \bold B)}.
  $$ 
  This can be computed in two other ways (1) reshsape the two matrices $\bold A$ and $\bold B$ as vectors, then take their usual dot product; and (2) $\text{tr}(\bold A^\top \bold B)$ which should be nice for the sake of theory, but makes *a lot* of unnecessary computation! The **Frobenius norm** is defined as 
  $$
  \lVert \bold A \rVert_F = \sqrt{\langle \bold A, \bold A\rangle_F} = \sqrt{\text{tr} (\bold A^\top \bold A)} = \sqrt{\sum\sum {a_{ij}}^2}.
  $$ 
    
    The fastest way to calculate this in NumPy is the straightforward `(A * B).sum()`. Other ways of calculating (shown in the video) are slower: (1) `np.dot(A.reshape(-1, order='F'), B.reshape(-1, order='F'))` where `order='F'` means Fortran-like indexing or along the columns, and (2) `np.trace(A @ B)`. 
    
    
    ```python
    In [14]: A = np.random.randn(2, 2)
    In [15]: B = np.random.randn(2, 2)
    In [17]: %timeit np.dot(A.reshape(-1, order='F'), B.reshape(-1, order='F'))
    5.57 µs ± 515 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    In [18]: %timeit np.trace(A.T @ B)
    7.79 µs ± 742 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    In [25]: %timeit (A * B).sum()
    3.73 µs ± 185 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    ```
    
    **Remark.** The Frobenius dot product is an inner product on $\mathbb R^{m \times n}$ in the same way that the usual dot product is an inner product on $\mathbb R^{mn}$. It follows that the Frobenius norm $\lVert \cdot \rVert_F$ is a norm as it is induced by the inner product $\langle \cdot, \cdot \rangle_F$ [[Prop. 6]](https://ai.stanford.edu/~gwthomas/notes/norms-inner-products.pdf). As usual for complex matrices we replace the transpose with the conjugate transpose: $\langle \bold A, \bold B\rangle_F =\text{tr}(\bold A^* \bold B)$ and $\lVert \bold A \rVert_F= \sqrt{\text{tr} (\bold A^* \bold A)} = \sqrt{\sum\sum |a_{ij}|^2}.$ These are an inner product and a norm on $\mathbb C^{m \times n}$, as in the real case.  <br><br>

* (4.60) **Other norms.** The **operator norm** is defined as $\lVert \bold A \rVert_p = \sup_{\bold x \neq \bold 0} \lVert \bold A \bold x \rVert_p / \lVert \bold x \rVert_p$ where we use the $p$-norm for vectors with $1 \leq p \leq \infty$. This just measures how much $\bold A$ scales the space, e.g. for isometries $\lVert \bold A \rVert_{p} = 1$. Another matrix norm, which unfortunately bears the same notation, is the **Schatten $p$-norm** defined as $\lVert \bold A  \rVert_p = \left( \sum_{i=1}^r \sigma_i^p \right)^{1/p}$ where $\sigma_1, \ldots, \sigma_r$ are the singular values of $\bold A$. That is, the Schatten $p$-norm is the $p$-norm of the vector of singular values of $\bold A$. Recall that the singular values are the length of the axes of the ellipse, so that Schatten $p$-norm is a cumulative measure of how much $\bold A$ expands the space around it in each dimension.
<br><br>

* (4.60) **Calculating the operator norm.** Here we approximate $\sup_{\lVert \bold x \rVert = 1} \lVert \bold A \bold x \rVert_p$ which is equivalent. Recall that the unit circle is transformed $\bold A$ to an ellipse whose length of axes are the singular values of $\bold A$. Thus, geometrically, we can guess that $\lVert \bold A \rVert_{p} = \sup_{\lVert \bold x \rVert = 1} \lVert \bold A \bold x \rVert_p = \sigma_1$ where $\sigma_1$ is the largest singular value of $\bold A$. We test this in `src/4_operator_norm.py` with the following results:
   ```
    approx:  1.8594375168610568
    numpy:   1.8594432066519955
    svd:     1.8594432066519957
   ```
    Awesome! Although, interestingly if we sample more than 100 equidistant points on the cicle, we get a worse approximation. (???)  




<!--- Template
* (4.46) **Punchline.**
  body.
  <br><br> 

* (4.46) **Test image.**<br>
  <img src="img/bb.jpg" alt="drawing" width="200"/>
  <br><br>
--->