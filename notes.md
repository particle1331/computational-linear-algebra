# Notes
  - (2.30) **No. of linearly independent vectors in ${\mathbb R^m}$**.  The maximum length $n$ of a list of linearly independent vectors in $\mathbb R^m$ is bounded by $m$.  If $n > m$, then the list is linearly dependent. <br><br>
  
  - (2.30) **Complexity of checking independence**.
  Suppose $n \leq m.$ What is the time complexity of showing n vectors in $\mathbb R^m$ are linearly independent? i.e. solving for nonzero solutions to $\bold A\bold x = \bold 0$. For instance, we have $\mathcal{O}(\frac{2}{3} mn^2)$ using Gaussian elimination assuming $\mathcal{O}(1)$ arithmetic which is a naive assumption as careless implementation can easily create numbers with with [exponentially many bits](https://cstheory.stackexchange.com/questions/3921/what-is-the-actual-time-complexity-of-gaussian-elimination)! In practice, the best way to compute the rank of $\bold A$ is through its SVD. This is, for example, how `numpy.linalg.matrix_rank` is implemented.
  <br><br>

  - (2.30) **Basis are non-unique.**
  Non-unique, but gives unique coordinate for each vector when the choice of basis is fixed. Some basis are better than others for a particular task, e.g. describing a dataset better. There are algorithms such as PCA & ICA that try to minimize some objective function.<br><br>

  * **Orthogonal matrices are precisely the linear isometries of $\mathbb R^n$.** A matrix $\bold A \in \mathbb R^n$ is an isometry if $\lVert \bold A \bold x\rVert^2 = \lVert \bold x \rVert^2$ for all $\bold x \in \mathbb R^n$. Note that $\lVert \bold A \bold x \rVert^2 = \bold x^\top\bold A^\top \bold A \bold x$ and $\lVert \bold x \rVert^2 = \bold x^\top \bold x$. So orthogonal matrices are isometries. Conversely, if a matrix $\bold A$ is an isometry, we can let $\bold x = \bold e_i - \bold e_j$ to get $\bold e_i^\top (\bold A^\top \bold A) \bold e_j = 
  (\bold A^\top \bold A)_{ij} 
  = \delta_{ij}$ where $\delta_{ij}$ is the Kronecker delta or $\bold A^\top\bold A = \bold I$. This tells us that length preserving matrices in $\mathbb R^n$ are necessarily orthogonal. Orthogonal matrices in $\mathbb R^2$ are either rotations or reflections &mdash; both of these are length preserving. The more surprising result is that these are the only length preserving matrices in $\mathbb R^2$!
  <br><br> 

  - **Orthogonal matrices as projections.** An orthogonal matrix $\bold U$ is defined as a matrix with orthonormal vectors in its column. It follows $\bold U^\top \bold U = \bold I.$ Since $\bold U$ is invertible, we can use uniqueness of inverse to get $\bold U \bold U^\top = \bold I.$ However, we can obtain this latter identity geometrically. Let $\bold x$ be a vector, then $\bold x = \sum_i \bold u_i \bold u_i^\top \bold x.$ This is true by uniqueness of components in a basis. Thus, $\bold U \bold U^\top \bold x = \bold x$ for any $\bold x,$ or $\bold U \bold U^\top  = \bold I.$ <br><br>

  - (3.34) **Shifting a matrix away from degeneracy:**
  $\bold A + \lambda \bold I = \tilde\bold A.$ 
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


* (4.51) **Geometry of linear operators.** In the code challenge, we saw that a unit circle is mapped by a square matrix $\bold A$ into an ellipse. It turns out that the effect of a square matrix $\bold A \in \mathbb R^{2 \times 2}$ as an operator on $\mathbb R^2$ is to dilate the space outwards in two orthogonal directions (possibly some directions shrinking to zero, but never in a negative direction), then resulting space is rotated twice. To see this, let $\bold A = \bold U \bold \Sigma \bold V^\top$ is the SVD of $\bold A$, then $\bold A = (\bold U \bold V^\top) (\bold V \bold\Sigma \bold V^\top)$. The factor $\bold V \bold\Sigma \bold V^\top$ dilates the space two orthogonal directions defined by the columns of $\bold V$ while the strength of the dilation is determined by the singular values in the diagonal of $\bold \Sigma$. 
We can interpret $\bold V$ and $\bold V^\top$ as change of basis matrices, i.e. in terms of a sum of projection operators $\sum_{i=1}^n \sigma_i \bold v_i \bold v_i^\top$. This is followed by a product $\bold U \bold V^\top$ of two isometries of $\mathbb R^2$. It can be [easily calculated](https://math.stackexchange.com/a/2924263) that orthogonal transformations of $\mathbb R^2$ are either rotations or reflections, so that we get a final ellipse. Note that the rank of $\bold A$ is equal to the number of nonzero singular values. So whenever $\bold A$ is singular, some of its singular values will be zero which results to a degenerate ellipse (see figure below). <br><br>

* (4.51) **Polar decomposition.** The decomposition of an operator $\bold A \in \mathbb R^{n\times n}$ in (4.51) into $\bold A = \bold Q \bold P$ where $\bold Q$ is orthogonal and $\bold P$ is symmetric positive semidefinite is called the **polar decomposition**. Geometrically, we can see that $\bold P$ should be unique. Indeed, observe that $\bold P^2 = \bold A^\top \bold A$ and $\bold A^\top \bold A$ is evidently symmetric positive semidefinite, so it has a unique symmetric positive semidefinite square root $\sqrt{\bold A^\top \bold A}$ [[Thm. 3]](https://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect7.pdf). Thus, $\bold P = \bold V \bold \Sigma \bold V^\top = \sqrt{\bold A ^\top \bold A}$ by uniqueness. Note however that the eigenvectors the orthogonal eigendecomposition into need not be unique (e.g. when the kernel of $\bold A$ is nonzero). For real matrices, the isometries are precisely the orthogonal matrices. Thus, the polar decomposition can be written as $\bold A = \bold Q \sqrt{\bold A^\top \bold A}$ for some isometry $\bold Q$ &mdash; cf. [[Lem. 9.6]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf) which states the polar decomposition in terms of the existence such an isometry. The matrix $\bold Q$ is only unique if $\bold A$ is nonsingular. For instance, if $\bold A$ is singular, then we can reflect across the axis where the space is collapsed and still get the same transformation. <br>
   
    **Remark.** The name "polar decomposition" comes from the analogous decomposition of complex numbers as $z = re^{i\theta}$ in polar coordinates. Here $r = \sqrt{\bar z z }$ (analogous to $\sqrt{\bold A^* \bold A}$) and multiplication by $e^{i\theta}$ is an isometry of $\mathbb C$ (analogous to the isometric property of $\bold Q$). For complex matrices we consider $\bold A^*\bold A$ and unitary matrices in the SVD. <br><br> 

* (4.51) **Computing the polar decomposition.** In `src\4_polar_decomp.py`, we verify the theory by calculating the polar decomposition from `u, s, vT = np.linalg.svd(A)`. We set `Q = u @ vT` and `P = vT.T @ np.diag(s) @ vT`. Some singular values are zero for singular `A`  (left) while all are nonzero for nonsingular `A` (right). The eigenvectors of `P` are scaled by the corresponding eigenvalues, then rotated with `Q`. The rotated eigenvectors of `P` lie along the major and minor axis of the ellipse: the directions where the circle is stretched prior to rotation. The code checks out in that the eigenvectors (obtained from SVD) line up nicely along the axes where the circle is elongated in the scatter plot (obtained by plotting the output vectors `A @ [x, y]` where `[x, y]` is a point on the unit circle). <br><br>
    <p align="center">
    <img src="img/4_polar_decomposition.png" title="drawing" width="600" />
    </p> <br>


* **SVD Proof.** The SVD states that any real matrix $\bold A \in \mathbb R^{m \times n}$ can be decomposed as $\bold A = \bold U \bold \Sigma \bold V^\top$ where $\bold U \in \mathbb R^{m \times m}$ and $\bold V \in \mathbb R^{n \times n}$ are orthonogonal matrices and $\bold\Sigma  \in \mathbb R^{m \times n}$ is a diagonal matrix with nonnegative real numbers on the diagonal. The diagonal entries $\sigma_i$ of $\bold \Sigma$ are called the **singular values** of $\bold A$. The number $r$ of nonzero singular values is equal to the rank of $\bold A$ as we will show shortly. 
<br><br>
The following proof of the SVD is constructive, i.e. we construct the singular values, and left and right singular vectors of $\bold A.$
Let $r = \text{rank }\bold A$, then $r \leq \min(m, n)$. 
Observe that $\bold A^\top \bold A \in \mathbb R^{n\times n}$ is symmetric positive semidefinite. 
It follows that the eigenvalues of $\bold A^\top \bold A$ are nonnegative. 
Thus, there exists an eigendecomposition $\bold A^\top \bold A = \bold V \bold \Sigma^2 \bold V^\top$ where $\bold V$ is an orthogonal matrix and $\bold \Sigma^2$ is a diagonal matrix of real eigenvalues of $\bold A^\top \bold A$ [[Theorem 8.3]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf). Here we let $\sigma_i = \bold \Sigma_{ii}$ such that $\sigma_1 \geq \sigma_2 \geq \ldots \sigma_r > 0.$
Here we use the fact that $\text{rank }\bold A ^\top \bold A = \text{rank }\bold A = r,$ and $\bold A^\top \bold A$ is similar to $\bold\Sigma^2,$ so that the first $r$ singular values of $\bold A$ are nonzero while the rest are zero. The singular values characterize the geometry of $\bold A$. For instance if $0 \leq r < m$, then the hyperellipse image of $\bold A$ collapses to have zero volume. The vectors $\bold v_1, \ldots, \bold v_n$ form an orthonormal basis for $\mathbb R^n$ which we call **right singular vectors.** Now that we are done with the domain of $\bold A,$ we proceed to its codomain.
<br><br> 
We know $\bold A \bold v_i$ for $i = 1, 2, \ldots, n$ span the image of $\bold A.$ For $i = 1, 2, \ldots, r$, it can be shown that $\lVert \bold A \bold v_i \rVert = \sigma_i.$ Since the first $r$ singular values are nonzero, we can define unit vectors $\bold u_i = {\sigma_i}^{-1}\bold A \bold v_i \in \mathbb R^m$ for $i = 1, \ldots, r.$ These are the **left singular vectors** of $\bold A.$ It follows that $\bold A \bold v_i = \sigma_i \bold u_i$ for $i = 1, \ldots, r.$ Observe that the vectors $\bold u_i$ are orthogonal 
  $$
  \bold u_i^\top \bold u_j = \frac{1}{\sigma_i\sigma_j}\bold v_i^\top\bold A^\top \bold A \bold v_j = \frac{1}{\sigma_i\sigma_j}\bold v_i^\top {\sigma_j}^2 \bold v_j = \delta_{ij} \frac{{\sigma_j}^2}{\sigma_i\sigma_j} = \delta_{ij}.
  $$
  Thus, $\bold u_1, \ldots \bold u_r$ is an orthonormal basis for the image of $\bold A$ in $\mathbb R^m.$ From here we can already obtain the **compact SVD** which already contains all necessary information: $\bold A = \sum_{k=1}^r \sigma_k \bold  u_k \bold v_k^\top$ or $\bold A = \bold U_r \bold \Sigma_r \bold V_r^\top$ (bottom of Figure 10.1 below where $r = n$) where $\bold U_r = [\bold u_1, \ldots, \bold u_r] \in \mathbb R^{m \times r},$ $\bold\Sigma_r = \bold\Sigma[:r, :r],$ and $\bold V^\top_r = \bold V[:, :r]^\top.$ To get the **full SVD**, we extend $\bold U_r$ to an orthonormal basis of $\mathbb R^m$ by Gram-Schmidt obtaining $\bold U = [\bold U_r | \bold U_{m-r}] \in \mathbb R^{m \times m}.$ For $\bold\Sigma$, we either pad ($m > n$) or remove zero rows ($m < n$) to get an $m \times n$ diagonal matrix. Finally, with these matrices, we can write $\bold A \bold V = \bold U \bold \Sigma$ so that $\bold A = \bold U \bold \Sigma \bold V^\top$ where the factors have the properties stated in the SVD. And we're done! $\square$

<br>

  <p align="center">
  <img src="img/svd.png" alt="drawing" width="400"/>
  </p>
  <br><br>

* **Singular vectors in the SVD.** Given the SVD we can write $\bold A = \sum_{i=1}^r \sigma_i \bold u_i \bold v_i^\top$ as a sum of rank one (!) terms. Recall that $\sigma_i \bold u_i = \bold A \bold v_i$. Writing $\bold A = \sum_{i=1}^r (\bold A \bold v_i) \bold v_i^\top$ is trivial given an ONB $\bold v_1, \ldots, \bold v_n$ of $\mathbb R^n.$ What is nontrivial in the SVD is that (1) an ONB always exists that is "natural" to $\bold A$, and (2) the corresponding image vectors $\bold A \bold v_i$ which span $\textsf{col }\bold A$ are also orthogonal in $\mathbb R^m.$
  <br>
  <p align="center">
  <img src="img/svd_ellipse.png" alt="Source: http://gregorygundersen.com/image/svd/ellipse.png" width="400"/>
  </p>
  <br><br>

  Another important characterization of the singular vectors is in terms of eigenvalues of $\bold A^\top \bold A$ and $\bold A \bold A^\top.$ By construction, $\bold v_1, \ldots, \bold v_n$ are eigenvectors of $\bold A^\top \bold A$ with respect to eigenvalues ${\sigma_1}^2, \ldots {\sigma_r}^2, 0, \ldots, 0$. On the other hand,
  $$
  \bold A \bold A^\top \bold u_i = \frac{1}{\sigma_i} \bold A \bold A^\top \bold A \bold v_i = \frac{1}{\sigma_i} {\sigma_i}^2 \bold A \bold v_i = {\sigma_i}^2 \bold u_i
  $$
  for $i = 1, \ldots, r.$ This is also trivially true for $i > r.$ Thus, $\bold u_1, \ldots, \bold u_m$ are $m$ orthogonal eigenvectors of $\bold A \bold A^\top$ with respect to eigenvalues ${\sigma_1}^2, \ldots {\sigma_r}^2, 0, \ldots, 0$.
  <br><br>


<br>
<p align="center">
  <img src="img/svd_change_of_basis.svg" alt="drawing" width="400"/> <br> 
  <b>Figure. </b> SVD as diagonalization.
</p>
<br>

* **SVD as diagonalization.** We can think of the SVD as a change of basis so that the $m \times n$ matrix $\bold A$ has a diagonal representation (see Figure above). Recall that we recover the components of a vector in an ONB by performing projection, so we can replace inverses with transpose. Let us see this in action. Recall that $\bold A \bold v_j = \sigma_j \bold u_j$ for $j \leq r$ and $\bold A \bold v_j = \bold 0$ for $r < j \leq n$. Then
    $$
    \begin{aligned}
    \bold A \bold x 
    &= \bold A \left(\sum_{j=1}^n \bold v_j \bold v_j^\top \bold x \right) \\
    &= \sum_{j=1}^n \left(\bold A \bold v_j \right) \bold v_j^\top \bold x  \\
    &= \sum_{i=1}^m \bold u_i \bold u_i^\top\sum_{j=1}^n \left(\bold A \bold v_j \right) \bold v_j^\top \bold x  \\
    &= \sum_{i=1}^m\sum_{j=1}^{\color{yellow} r}  \bold u_i \bold u_i^\top \left(\sigma_{j} \bold u_j \right) \bold v_j^\top \bold x  \\
    &= \sum_{i=1}^m\sum_{j=1}^{\color{yellow} n}  \bold u_i \bold u_i^\top (\Sigma_{ij} \bold u_i ) \bold v_j^\top \bold x  \\
    &= \sum_{i=1}^m \sum_{j=1}^n  \bold u_i  \Sigma_{ij} \bold v_j^\top \bold x  \\
    \end{aligned}
    $$
    The entry $\Sigma_{ij}$ is equal to $\bold u_i^\top \bold A \bold v_j$ where $\bold u_i$ and $\bold v_j$ are ONBs. Thus, the SVD is analogous to diagonalization for square matrices, but instead of eigenvalues, we diagonalize into an $m \times n$ diagonal matrix of singular values. From [Chapter 10](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/eigs.pdf) of Moler's *Numerical Computing with MATLAB*:
  > In abstract linear algebra terms, eigenvalues are relevant if a square, $n$-by-$n$ matrix $\bold A$ is thought of as mapping $n$-dimensional space onto itself. We try to find a basis for the space so that the matrix becomes diagonal. This basis might be complex even if $\bold A$ is real. In fact, if the eigenvectors are not linearly independent, such a basis does not even exist. The SVD is relevant if a possibly rectangular, $m$-by-$n$ matrix $\bold A$ is thought of as mapping $n$-space onto $m$-space. We try to find one change of basis in the domain and a usually different change of basis in the range so that the matrix becomes diagonal. Such bases always exist and are always real if $\bold A$ is real. In fact, the transforming matrices are orthogonal or unitary, so they preserve lengths and angles and do not magnify errors.

<br>

* **Computing the SVD.** In `4_compute_svd.py` we calculate 3 things: (1) equality between the eigenvalues of $\sqrt{\bold A^\top \bold A}$ and the singular values of $\bold A$, (2) difference bet. max. singular value $\sigma_1$ and $\max_{\lVert \bold x \rVert_2 = 1} \lVert \bold A \bold x \rVert_2$, and (3) whether $\bold A\bold v_i = \sigma_i \bold u_i$ for $i = 1, 2$. Here $\bold A$ is a 2x2 matrix with elements sampled from a standard normal.
  ```
    eigvals of sqrt(A^T A):  [1.29375301 2.75276951]
    singular values of A:    [2.75276951 1.29375301]
    max norm - max s.value:  1.6732994501111875e-07
    || Av - su ||.max():     2.220446049250313e-16
  ```
  <br>

* **A nondiagonalizable matrix.** The matrix $\bold A = [[1, 0], [1, 1]]$ has eigenvalues are $\lambda_1 = \lambda_2 = 1$ with eigenvectors of the form $\bold v = [0, t]^\top$ for nonzero $t \in \mathbb R$. It follows that $\bold A$ is not diagonalizable since it has at most one linearly independent eigenvectors &mdash; not enough to span $\mathbb R^2.$ <br><br>

* (4.56) **Symmetric product of two symmetric matrices.** Suppose $\bold S$ and $\bold T$ are symmetric matrices. What is the condition so that their product $\bold S \bold T$ is symmetric? Observe that $(\bold S \bold T)^\top = \bold T ^\top \bold S ^\top = \bold T \bold S.$ Thus, the product of two symmetric matrices is symmetric if and only if the matrices commute. This works for a very small class of matrices, such as zeros or constant diagonal matrices. 
In the case of $2 \times 2$ matrices, this is satisfied most naturally by  matrices with constant diagonal entries &mdash; this is just a quirk that does not generalize to higher dimensional matrices.
The lack of symmetry, i.e. $\bold S \bold T \neq \bold T \bold S$, turns out to be extremely important in machine-learning, multivariate statistics, and signal processing, and is a core part of the reason why linear classifiers are so successful [[Lec 56, Q&A]](https://www.udemy.com/course/linear-algebra-theory-and-implementation/learn/lecture/10738628#questions/13889570/): 
    >  "The lack of symmetry means that $\bold C=\bold B^{-1} \bold A$ is not symmetric, which means that $\bold C$ has non-orthogonal eigenvectors. In stats/data science/ML, most linear classifiers work by using generalized eigendecomposition on two data covariance matrices $\bold B$ and $\bold A$, and the lack of symmetry in $\bold C$ turns a compression problem into a separation problem."

    (???)<br><br>

* (4.57) **Hadamard and standard multiplications are equivalent for diagonal matrices.** This can have consequences in practice. The following code in IPython shows that Hadamard multiplication is 3 times faster than standard multiplication in NumPy.
    ```
    In [1]: import numpy as np
    In [2]: D = np.diag([1, 2, 3, 4, 5])
    In [3]: %timeit D @ D
    2.21 µs ± 369 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    In [4]: %timeit D * D
    717 ns ± 47.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    ```     
<br>


* (4.59) **Frobenius inner product and its induced norm.** The **Frobenius inner product** between two $m \times n$ matrices $\bold A$ and $\bold B$ is defined as 
  $
  \langle \bold A, \bold B\rangle_F 
  = \sum_{i=1}^m \sum_{j=1}^n a_{ij} b_{ij}. 
  $ 
  Two alternative ways of computing this: (1) reshape $\bold A$ and $\bold B$ as vectors, then take the dot product; and (2) $\langle \bold A, \bold B\rangle_F = \text{tr}(\bold A^\top \bold B)$ which is nice in theory, but makes *a lot* of unnecessary computation! The **Frobenius norm** is defined as 
  $$
  \lVert \bold A \rVert_F = \sqrt{\langle \bold A, \bold A\rangle_F} = \sqrt{\text{tr} (\bold A^\top \bold A)} = \sqrt{\sum\sum {a_{ij}}^2}.
  $$ 
    The fastest way to calculate this in NumPy is the straightforward `(A * B).sum()`. Other ways of calculating (shown in the video) are slower: (1) `np.dot(A.reshape(-1, order='F'), B.reshape(-1, order='F'))` where `order='F'` means Fortran-like indexing or along the columns, and (2) `np.trace(A @ B)`. 
    ```
    In [14]: A = np.random.randn(2, 2)
    In [15]: B = np.random.randn(2, 2)
    In [17]: %timeit np.dot(A.reshape(-1, order='F'), B.reshape(-1, order='F'))
    5.57 µs ± 515 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    In [18]: %timeit np.trace(A.T @ B)
    7.79 µs ± 742 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    In [25]: %timeit (A * B).sum()
    3.73 µs ± 185 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    ```
    **Remark.** The Frobenius inner product is an inner product on $\mathbb R^{m \times n}$ in the same way that the usual dot product is an inner product on $\mathbb R^{mn}$. It follows that the Frobenius norm $\lVert \cdot \rVert_F$ is a norm as it is induced by the inner product $\langle \cdot, \cdot \rangle_F$ [[Prop. 6]](https://ai.stanford.edu/~gwthomas/notes/norms-inner-products.pdf). As usual for complex matrices we replace the transpose with the conjugate transpose: $\langle \bold A, \bold B\rangle_F =\text{tr}(\bold A^* \bold B)$ and $\lVert \bold A \rVert_F= \sqrt{\text{tr} (\bold A^* \bold A)} = \sqrt{\sum\sum |a_{ij}|^2}.$ These are an inner product and a norm on $\mathbb C^{m \times n}$, as in the real case.  <br><br>


* (4.60) **Other norms.** The **operator norm** is defined as $\lVert \bold A \rVert_p = \sup_{\bold x \neq \bold 0} \lVert \bold A \bold x \rVert_p / \lVert \bold x \rVert_p$ where we use the $p$-norm for vectors with $1 \leq p \leq \infty$. This just measures how much $\bold A$ scales the space, e.g. for isometries $\lVert \bold A \rVert_{p} = 1$. Another matrix norm, which unfortunately bears the same notation, is the **Schatten $p$-norm** defined as $\lVert \bold A  \rVert_p = \left( \sum_{i=1}^r \sigma_i^p \right)^{1/p}$ where $\sigma_1, \ldots, \sigma_r$ are the singular values of $\bold A$. That is, the Schatten $p$-norm is the $p$-norm of the vector of singular values of $\bold A$. Recall that the singular values are the length of the axes of the ellipse, so that Schatten $p$-norm is a cumulative measure of how much $\bold A$ expands the space around it in each dimension.
<br><br>

* (4.60) **Operator norm and singular values.** Note that $\lVert \bold A \rVert_{p} = \sup_{\lVert \bold x \rVert = 1} \lVert \bold A \bold x \rVert_p$ for the operator norm. Recall that the unit circle is transformed $\bold A$ to an ellipse whose axes have length corresponding to the singular values of $\bold A$. Geometrically, we can guess that $\lVert \bold A \rVert_{p} = \max_{j} \sigma_j$  where $\sigma_j$ are the singular values of $\bold A$. Indeed, we can verify this:
    ```
    In [6]: A = np.random.randn(2, 2)
    In [7]: abs(np.linalg.norm(A, 2) - np.linalg.svd(A)[1][0])
    Out[7]: 0.0
    ```

<br>

* (5.64) **Rank as dimensionality of information.** The rank of $\bold A$ is the number of maximal linearly independent columns of $\bold A.$ Any matrix $\bold A$ has a rank $r \in \mathbb N$ such that $0 \leq r \leq \min(m, n).$ Matrix rank has several applications, e.g. $\bold A^{-1}$ exists for a square matrix whenever it has maximal rank. In applied settings, rank is used in PCA, Factor Analysis, etc. because rank is used to determine how much nonredundant information is contained in $\bold A.$ <br><br>

* (5.65) **Computing the rank.** How to count the maximal number of linearly independent columns? (1) Row reduction (can be numerically unstable). (2) Best way is to use SVD. The rank of $\bold A$ is the number $r$ of nonzero singular values of $\bold A.$ This is how it's implemented in MATLAB and NumPy. The SVD is also used for rank estimation. Another way is to count the number of nonzero eigenvalues of $\bold A$ provided (!) $\bold A$ has an eigendecomposition. Since this would imply that $\bold A$ is similar to its matrix of eigenvalues. This is in general not true. Instead, we can count the eigenvalues of $\bold A^\top \bold A$ or $\bold A\bold A^\top$ &mdash; whichever is smaller &mdash; since an eigendecomposition for both matrices always exist. <br><br>

* (5.65) **Rank can be difficult to calculate numerically.** For instance if we obtain $\sigma_k = 10^{-13}$ numerically, is it a real nonzero singular value, or is it zero? In practice, we set thresholds. The choice of threshold can be arbitrary or domain specific, and in general, introduces its own difficulties. Another issue is noise, adding $\epsilon\bold I$ makes $\bold A = [[1, 1], [1, 1]]$ rank two. <br><br>

* (5.67) **Generate rank 4 matrix 10x10 matrix randomly by multiplying two randomly generated matrices.** Solution is to multiply 10x4 and 4x10 matrices. Here it is assumed that the randomly generated matrices have maximal rank. <br><br>

* (5.69) **Rank of $\bold A^\top \bold A$ and $\bold A \bold A^\top$.** These are all equal to the rank of $\bold A.$ 
The first equality can be proved using by showing the $\ker \bold A^\top \bold A = \ker \bold A,$ then invoke the rank-nullity theorem. 
Now that we know this is true, we can now use the SVD to find the rank of $\bold A \bold A^\top$ which results in $\bold A \bold A^\top = \bold U \bold \Sigma^2 \bold U^\top,$ i.e. similar to a diagonal matrix with $r = \text{rank }\bold A$ diagonal entries. 
Thus, $\text{rank } \bold A \bold A^\top = \text{rank }\bold A = r.$ <br><br>

* (5.71) **Making a matrix full-rank by shifting:** $\tilde\bold A = \bold A + \lambda \bold I$ where we assume $\bold A$ is square. This is done for computational stability reasons. Typically the regularization constant $\lambda$ is less than the experimental noise. For instance, if $|\lambda| \gg \max |a_{ij}|,$ then $\tilde \bold A \approx \lambda \bold I$ and $\bold A$ becomes the noise. An exchange in the Q&A highlights another important issue. Hamzah asks:
  > So in a previous video in this section, you talked about how a 3 dimensional matrix spanning a 2 dimensional subspace [...] really is a rank 2 matrix, BUT if you introduce some noise, it can look like like a rank 3 matrix. [...] By adding the identity matrix, aren't you essentially deliberately adding noise to an existing dataset to artificially boost the rank? Am I correct in interpreting that you can possibly identify features in the boosted rank matrix that may not actually exist in the true dataset, and maybe come up with some weird conclusions? If that is the case wouldn't it be very dangerous to increase the rank by adding the identity matrix? Would appreciate some clarification. Thank you!

  To which Mike answers:

  > Excellent observation, and good question. Indeed, there is a huge and decades-old literature about exactly this issue -- how much "noise" to add to data? In statistics and machine learning, adding noise is usually done as part of regularization. <br><br>
  The easy answer is that you want to shift the matrix by as little as possible to avoid changing the data, while still adding enough to make the solutions work. I don't go into a lot of detail about that in this course, but often, somewhere around 1% of the average eigenvalues of the matrix provides a good balance. <br><br>
  Note that this is done for numerical stability reasons, not for theoretical reasons. So the balance is: Do I want to keep my data perfect and get no results, or am I willing to lose a bit of data precision in order to get good results?

<br>

* (5.72) **Is this vector in the span of this set?** Let $\bold x \in \mathbb R^m$ be a test vector. Is $\bold x$ in the span of $\bold a_1, \ldots, \bold a_n \in \mathbb R^m.$ Let $\bold A = [\bold a_1, \ldots, \bold a_n]$ with rank $r.$ The solution is to check whether the rank of $[\bold A | \bold x]$ is equal to the $r$ (in span) or $r+1$ (not in span). <br><br>

* 

<!--- Template
* (4.46) **Punchline.**
  body.
  <br><br> 

* (4.46) **Test image.**<br>
  <p align="center">
  <img src="img/bb.jpg" alt="drawing" width="200"/>
  </p>
  <br><br>
--->