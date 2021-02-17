# Notes
  - (2.30) **No. of linearly independent vectors in ${\mathbb R^m}$**.  The maximum length $n$ of a list of linearly independent vectors in $\mathbb R^m$ is bounded by $m$.  If $n > m$, then the list is linearly dependent. <br><br>
  
  - (2.30) **Complexity of checking independence**.
  Suppose $n \leq m.$ What is the time complexity of showing n vectors in $\mathbb R^m$ are linearly independent? i.e. solving for nonzero solutions to $\bold A\bold x = \bold 0$. For instance, we have $\mathcal{O}(\frac{2}{3} mn^2)$ using Gaussian elimination assuming $\mathcal{O}(1)$ arithmetic which is a naive assumption as careless implementation can easily create numbers with with [exponentially many bits](https://cstheory.stackexchange.com/questions/3921/what-is-the-actual-time-complexity-of-gaussian-elimination)! In practice, the best way to compute the rank of $\bold A$ is through its SVD. This is, for example, how `numpy.linalg.matrix_rank` is implemented.
  <br><br>

  - (2.30) **Basis is non-unique.**
  A choice of basis is non-unique but gives unique coordinates for each vector once the choice of basis is fixed. Some basis are better than others for a particular task, e.g. describing a dataset better. There are algorithms such as PCA & ICA that try to minimize some objective function.<br><br>

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


* (4.51) **Geometry of linear operators.** In the code challenge, we saw that a unit circle is mapped by a square matrix $\bold A$ into an ellipse. It turns out that the effect of a square matrix $\bold A \in \mathbb R^{2 \times 2}$ as an operator on $\mathbb R^2$ is to dilate the space outwards in two orthogonal directions (possibly some directions shrinking to zero, but never in a negative direction), then resulting space is rotated twice. To see this, let $\bold A = \bold U \bold \Sigma \bold V^\top$ be the SVD of $\bold A$, then $\bold A = (\bold U \bold V^\top) (\bold V \bold\Sigma \bold V^\top)$. The factor $\bold V \bold\Sigma \bold V^\top$ dilates the space two orthogonal directions defined by the columns of $\bold V$ while the strength of the dilation is determined by the singular values in the diagonal of $\bold \Sigma$. 
We can interpret $\bold V$ and $\bold V^\top$ as change of basis matrices, i.e. in terms of a sum of projection operators $\sum_{i=1}^n \sigma_i \bold v_i \bold v_i^\top$. This is followed by a product $\bold U \bold V^\top$ of two isometries of $\mathbb R^2$. It can be [easily calculated](https://math.stackexchange.com/a/2924263) that orthogonal transformations of $\mathbb R^2$ are either rotations or reflections, so that we get a final ellipse. Since the rank of $\bold A$ is equal to the number of nonzero singular values, whenever $\bold A$ is singular, some of its singular values will be zero corresponding to an axis where the ellipse collapses (see figure below). <br><br>

* (4.51) **Polar decomposition.** The decomposition of an operator $\bold A \in \mathbb R^{n\times n}$ in (4.51) into $\bold A = \bold Q \bold P$ where $\bold Q$ is orthogonal and $\bold P$ is symmetric positive semidefinite is called the **polar decomposition**. Geometrically, we can see that $\bold P$ should be unique. Indeed, observe that $\bold P^2 = \bold A^\top \bold A$ and $\bold A^\top \bold A$ is evidently symmetric positive semidefinite, so it has a unique symmetric positive semidefinite square root $\sqrt{\bold A^\top \bold A}$ [[Thm. 3]](https://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect7.pdf). Thus, $\bold P = \bold V \bold \Sigma \bold V^\top = \sqrt{\bold A ^\top \bold A}$ by uniqueness. Note however that the eigenvectors the orthogonal eigendecomposition into need not be unique (e.g. when the kernel of $\bold A$ is nonzero). For real matrices, the isometries are precisely the orthogonal matrices. Thus, the polar decomposition can be written as $\bold A = \bold Q \sqrt{\bold A^\top \bold A}$ for some isometry $\bold Q$; cf. [[Lem. 9.6]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf) which states the polar decomposition in terms of the existence of such an isometry. The matrix $\bold Q$ is only unique if $\bold A$ is nonsingular. For instance, if $\bold A$ is singular, then we can reflect across the axis where the space is collapsed and still get the same transformation. <br>
   
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
and that there exists an eigendecomposition $\bold A^\top \bold A = \bold V \bold \Sigma^2 \bold V^\top$ where $\bold V$ is an orthogonal matrix and $\bold \Sigma^2$ is a diagonal matrix of real eigenvalues of $\bold A^\top \bold A$ [[Theorem 8.3]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf). Here we let $\sigma_i = \bold \Sigma_{ii}$ such that $\sigma_1 \geq \sigma_2 \geq \ldots \sigma_r > 0$ where $r = \text{rank }\bold A.$ This comes from $\text{rank }\bold A ^\top \bold A = \text{rank }\bold A = r,$ and $\bold A^\top \bold A$ is similar to $\bold\Sigma^2,$ so that the first $r$ singular values of $\bold A$ are nonzero while the rest are zero. The singular values characterize the geometry of $\bold A$. For instance if $0 \leq r < m$, then the hyperellipse image of $\bold A$ collapses to have zero volume. The vectors $\bold v_1, \ldots, \bold v_n$ form an orthonormal basis for $\mathbb R^n$ which we call **right singular vectors.** Now that we are done with the domain of $\bold A,$ we proceed to its codomain.
<br><br> 
We know $\bold A \bold v_i$ for $i = 1, 2, \ldots, n$ span the image of $\bold A.$ For $i = 1, 2, \ldots, r$, it can be shown that $\lVert \bold A \bold v_i \rVert = \sigma_i.$ Since the first $r$ singular values are nonzero, we can define unit vectors $\bold u_i = {\sigma_i}^{-1}\bold A \bold v_i \in \mathbb R^m$ for $i = 1, \ldots, r.$ These are the **left singular vectors** of $\bold A.$ It follows that $\bold A \bold v_i = \sigma_i \bold u_i$ for $i = 1, \ldots, r$ and $\bold A \bold v_i = \bold 0$ for $i > r.$ Observe that the vectors $\bold u_i$ are orthogonal 
  $$
  \bold u_i^\top \bold u_j = \frac{1}{\sigma_i\sigma_j}\bold v_i^\top\bold A^\top \bold A \bold v_j = \frac{1}{\sigma_i\sigma_j}\bold v_i^\top {\sigma_j}^2 \bold v_j = \delta_{ij} \frac{{\sigma_j}^2}{\sigma_i\sigma_j} = \delta_{ij}.
  $$
  Thus, $\bold u_1, \ldots \bold u_r$ is an orthonormal basis for the image of $\bold A$ in $\mathbb R^m.$ From here we can already obtain the **compact SVD** which already contains all necessary information: $\bold A = \sum_{k=1}^r \sigma_k \bold  u_k \bold v_k^\top$ or $\bold A = \bold U_r \bold \Sigma_r \bold V_r^\top$ (bottom of Figure 10.1 below with $r = n$) where $\bold U_r = [\bold u_1, \ldots, \bold u_r] \in \mathbb R^{m \times r},$ $\bold\Sigma_r = \bold\Sigma[:r, :r],$ and $\bold V^\top_r = \bold V[:, :r]^\top.$ To get the **full SVD**, we extend $\bold U_r$ to an orthonormal basis of $\mathbb R^m$ by Gram-Schmidt obtaining $\bold U = [\bold U_r | \bold U_{m-r}] \in \mathbb R^{m \times m}.$ For $\bold\Sigma$, we either pad ($m > n$) or remove zero rows ($m < n$) to get an $m \times n$ diagonal matrix. Finally, with these matrices, we can write $\bold A \bold V = \bold U \bold \Sigma$ so that $\bold A = \bold U \bold \Sigma \bold V^\top$ where the factors have the properties stated in the SVD. And we're done! $\square$

<br>
  <p align="center">
  <img src="img/svd.png" alt="drawing" width="400"/>
  </p>
<br>

<br>

* See `src/4_svd_from_scratch.py` for a construction of the (compact) SVD in code following the proof. The result looks great:
    <br>

  ```python
  In [1]: %run 4_svd_from_scratch.py                                                                                                                                    
  U @ Sigma @ V.T =
  [[ 1.76405235  0.40015721  0.97873798  2.2408932 ]
   [ 1.86755799 -0.97727788  0.95008842 -0.15135721]
   [-0.10321885  0.4105985   0.14404357  1.45427351]
   [ 0.76103773  0.12167502  0.44386323  0.33367433]
   [ 1.49407907 -0.20515826  0.3130677  -0.85409574]]

  A=
  [[ 1.76405235  0.40015721  0.97873798  2.2408932 ]
   [ 1.86755799 -0.97727788  0.95008842 -0.15135721]
   [-0.10321885  0.4105985   0.14404357  1.45427351]
   [ 0.76103773  0.12167502  0.44386323  0.33367433]
   [ 1.49407907 -0.20515826  0.3130677  -0.85409574]]

  L1 error = 8.229528170033973e-15

  U.T @ U =
  [[ 1.00000000e+00  7.85139767e-17 -2.70399175e-16  4.08696943e-15 0.00000000e+00]
   [ 7.85139767e-17  1.00000000e+00 -9.81442401e-16  6.71377649e-16 0.00000000e+00]
   [-2.70399175e-16 -9.81442401e-16  1.00000000e+00  1.27931484e-14 0.00000000e+00]
   [ 4.08696943e-15  6.71377649e-16  1.27931484e-14  1.00000000e+00 0.00000000e+00]
   [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00]]
  ```

<br>

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
  for $i = 1, \ldots, r.$ This is also trivially true for $i > r.$ Thus, $\bold u_1, \ldots, \bold u_m$ are $m$ orthogonal eigenvectors of $\bold A \bold A^\top$ w.r.t. eigenvalues ${\sigma_1}^2, \ldots {\sigma_r}^2, 0, \ldots, 0$.
  <br><br>

<br>
<p align="center">
  <img src="img/svd_change_of_basis.svg" alt="drawing" width="400"/> <br> 
  <b>Figure. </b> SVD as diagonalization.
</p>
<br>

* **SVD as diagonalization.** We can think of the SVD as a change of basis so that the $m \times n$ matrix $\bold A$ has a diagonal representation (see Figure above). Recall that we recover the components of a vector in an ONB by performing projection, so we can replace inverses with transpose. In action: $\bold A = \bold U \bold U^\top \bold A \bold V \bold V^\top = \bold U \bold \Sigma \bold V^\top.$ Here $\bold U \bold U^\top = \sum_{i = 1}^m \bold u_i \bold {u_i}^\top$ is the change of basis of output vectors of $\bold \Sigma$ defined by the columns of $\bold U$ and, similarly, $\bold V \bold V^\top = \sum_{j = 1}^m \bold v_j \bold {v_j}^\top$ is the change of basis of input vectors of $\bold \Sigma$ defined by ONB of $\mathbb R^n$ that form the columns of $\bold V.$ Thus, the SVD is analogous to diagonalization for square matrices, but instead of eigenvalues, we diagonalize into an $m \times n$ diagonal matrix of singular values. From [Chapter 10](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/eigs.pdf) of Moler's *Numerical Computing with MATLAB*: <br><br>
  > In abstract linear algebra terms, eigenvalues are relevant if a square, $n$-by-$n$ matrix $\bold A$ is thought of as mapping $n$-dimensional space onto itself. We try to find a basis for the space so that the matrix becomes diagonal. This basis might be complex even if $\bold A$ is real. In fact, if the eigenvectors are not linearly independent, such a basis does not even exist. The SVD is relevant if a possibly rectangular, $m$-by-$n$ matrix $\bold A$ is thought of as mapping $n$-space onto $m$-space. We try to find one change of basis in the domain and a usually different change of basis in the range so that the matrix becomes diagonal. Such bases always exist and are always real if $\bold A$ is real. In fact, the transforming matrices are orthogonal or unitary, so they preserve lengths and angles and do not magnify errors.

<br>

* **Computing the SVD.** In `4_compute_svd.py` we calculate 3 things: (1) equality between the eigenvalues of $\sqrt{\bold A^\top \bold A}$ and the singular values of $\bold A$, (2) difference bet. max. singular value $\sigma_1$ and $\max_{\lVert \bold x \rVert_2 = 1} \lVert \bold A \bold x \rVert_2$, and (3) whether $\bold A\bold v_i = \sigma_i \bold u_i$ for $i = 1, 2$. Here $\bold A$ is a 2x2 matrix with elements sampled from a standard normal.
  ```python
    eigvals of sqrt(A^T A):  [1.29375301 2.75276951]
    singular values of A:    [2.75276951 1.29375301]
    max norm - max s.value:  1.6732994501111875e-07
    || Av - su ||.max():     2.220446049250313e-16
  ```
  <br>

* **Spectral theorem proof.** The spectral theorem is an extremely beautiful result which one can think of as the SVD for linear operators. In fact, the construction of the SVD relies on a spectral decomposition, i.e. of $\bold A^\top \bold A$ which is automatically symmetric. 
A key property of symmetric matrices used in the proof is that if $V$ is a subspace, then $V^\perp$ is invariant under $\bold A.$ This will allow us to recursively construct the eigenvector directions of $\bold A.$ The real spectral theorem generalizes to self-adjoint operators on real inner product spaces as in [[Theorem 8.3]](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf). <br><br>

    > **Theorem.** (Real spectral theorem). Let $\bold A \in \mathbb R^{n \times n}$ be a symmetric matrix. Then
      (1) the eigenvalues of $\bold A$ are real;
      (2) the eigenvectors of $\bold A$ corresponding to distinct eigenvalues are orthogonal; and
      (3) there exists an ONB of $\mathbb R^n$ of eigenvectors of $\bold A.$ This allows the diagonalization 
      $\bold A = \sum_{k=1}^n \lambda_k \bold v_k {\bold v_k}^\top = \bold V \bold \Lambda \bold V^\top$ 
      where $\bold V$ is a real orthogonal matrix of column stacked eigenvectors $\bold v_1, \ldots, \bold v_n$ and $\bold \Lambda$ is a real diagonal matrix of eigenvalues $\lambda_1, \ldots, \lambda_n.$ 
  
  <br>
  
  **Proof.** [Olver, 2018]. We skip (1) and (2). To prove (3), we perform induction on $n.$ For $n = 1$, this is trivially true with $\bold A = [a]$ and $\lambda = a \in \mathbb R$ with eigenvector $1.$ Suppose $n \geq 2$ and the spectral theorem is true for symmetric matrices in $\mathbb R^{n-1}.$ By the [Fundamental Theorem of Algebra (FTA)](https://math.libretexts.org/Bookshelves/Linear_Algebra/Book%3A_Linear_Algebra_(Schilling_Nachtergaele_and_Lankham)/07%3A_Eigenvalues_and_Eigenvectors/7.04%3A_Existence_of_Eigenvalues), there exists at least one eigenvalue $\lambda$ of $\bold A$ which we know to be real. Along with $\lambda$ comes a nonzero unit eigenvector $\bold v \in \mathbb R^n.$ Let $\bold v^\perp$ be the subspace orthogonal to the $1$-dimensional subspace spanned by $\bold v.$ Then, $\dim (\bold v^\perp) = n-1$ so that $\bold v^\perp$ has an orthonormal basis $\bold y_1, \ldots, \bold y_{n-1} \in \mathbb R^n.$ Moreover, $\bold v^\perp$ is invariant under $\bold A$ as a consequence of symmetry.
  Suppose $\bold w \in \bold v^\perp,$ then 
  $$
  \begin{aligned}
  (\bold A \bold w)^\top \bold v 
  &= \bold w ^\top \bold A^\top \bold v  \\
  &= \bold w ^\top \bold A \bold v \\
  &= \lambda \bold w ^\top \bold v = 0.
  \end{aligned}
  $$
  That is, $\bold A \bold w \in \bold v^\perp.$ It follows that the restriction ${\bold A}{|_ {\bold v^\perp}}$ of $\bold A$ on $\bold v^\perp$ is well-defined and we can write $\bold A| _ {\bold v^\perp} = \bold Y \bold B \bold Y^\top$ where $\bold Y = [\bold y_1, \ldots , \bold y_{n-1}] \in \mathbb R ^{n \times (n-1)}$ and $\bold B \in \mathbb R^{(n-1) \times (n-1)}$ is the coordinate representation of $\bold A| _ {\bold v^\perp},$ i.e. $\bold B = \bold Y^\top \bold A \bold Y.$ Observe that $\bold B$ is symmetric:
  $$
  b_{ij} = {\bold y_i}^\top \bold A \bold y_j = (\bold A^\top \bold y_i)^\top \bold y_j = (\bold A \bold y_i)^\top \bold y_j = b_{ji}.
  $$
  By the inductive hypothesis, $\bold B$ has a spectral decomposition in terms of real eigenvalues $\omega_1, \ldots, \omega_{n-1}$ and orthonormal eigenvectors $\bold u_1, \ldots, \bold u_{n-1}$ so that $\bold B = \bold U \bold \Omega \bold U^\top$ where $\bold \Omega = \text{diag}(\omega_1, \ldots, \omega_{n-1})$ is a diagonal matrix of real eigenvalues $\omega_1, \ldots, \omega_{n-1}$ and $\bold U = [\bold u_1, \ldots, \bold u_{n-1}] \in \mathbb R^{(n-1) \times (n-1)}$ is orthogonal. Thus, 
  $$
  {\bold A}{|_ {\bold v^\perp}} = (\bold Y \bold U) \bold \Omega ( \bold Y \bold U)^\top.
  $$
  Let $\bold w_ j = \sum_{k=1}^{n-1} u_{kj} \bold y_k = \bold Y \bold u_j \in \bold v^\perp$ for $j = 1, \ldots, n-1.$ 
  We use the amazing fact that the inner product of vectors $\bold w_i$ and $\bold w_j$ represented under an ONB $\bold y_1, \ldots, \bold y_{n-1}$ reduces to the inner product of its coordinate vectors $\bold u_i$ and $\bold u_j$ which are orthonormal by the inductive hypothesis!  That is,
  $$
  {\bold w_ i}^\top \bold w_j = {(\bold Y \bold u_i)}^\top {\bold Y \bold u_j} = {\bold u_i}^\top \bold Y ^\top {\bold Y \bold u_j} = \delta_{ij}.
  $$
  Hence, $\bold w_1, \ldots, \bold w_{n-1}$ is an ONB for $\bold v^\perp.$ Since $\bold v \perp \bold w_j$ for $j=1, \ldots, n-1,$ by maximality (1) $\bold v, \bold w_1 \ldots, \bold w_{n-1}$ is an orthonormal basis of $\mathbb R^n.$ 
  Furthermore, (2) $\bold A \bold v = \lambda \bold v$ and $\bold A \bold w_j = \omega_j \bold w_j$ for $j=1, \ldots, n-1.$ 
  These two facts allows us to write
  $$
  \begin{aligned}
  \bold A
  &= \lambda \bold v \bold v^\top + \sum_{j=1}^{n-1}\omega_j \bold w_j{\bold w_j}^\top \\
  &= \Bigg[\bold v\; \bold w_1 \ldots \; \bold w_{n-1}\Bigg] \begin{bmatrix}
   \lambda & & \\ 
     & \omega_1 & & \\ 
     &   &  \ddots & \\
     &   &  & \omega_{n-1}
  \end{bmatrix}
  \begin{bmatrix}
  \bold v^\top \\
  {\bold w_1}^\top \\ 
  \vdots
  \\
  {\bold w_{n-1}}^\top
  \end{bmatrix}.
  \end{aligned}
  $$
  Observe that (1) allowed a coordinate representation $\bold A = \bold V \bold \Omega \bold V^\top$ where $\bold V$ is orthogonal, and (2) guaranteed that $\bold \Omega$ is diagonal. 
  This completes the proof! $\square$ 
  
<br>

* **Code demo: Spectral theorem proof.** In `4_spectral_theorem.py` we implement the constuction above of an orthonormal eigenbasis for $\mathbb R^n$ for $n = 3$ with respect to a randomly generated symmetric matrix `A`. The first eigenvector $\bold v$ is obtained by cheating a bit, i.e. using `np.linalg.eig`. Then, two linearly independent vectors $\bold y_1$ and $\bold y_2$ were constructed by calculating the equation of the plane orthogonal to $\bold v$ and finding $x$'s such that $(x, 1, 1)$ and $(x, 1, 0)$ are points on the plane $\bold v^\perp.$ Finally, the vectors $\bold y_1$ and $\bold y_2$ are made to be orthonormal by Gram-Schmidt. By the inductive hypothesis, we are allowed to compute `omega, U = np.linalg.eig(B)` where `B = Y.T @ A @ Y`. Then, we set `W = Y @ U` to be the $n-1$ eigenvector directions in the orthogonal plane. This is concatenated with $\bold v$ to get the final matrix `V` of all $n$ eigenvectors. The eigenvalues are constructed likewise in decreasing order. 

  <br>

  ```python
  In [123]: %run 4_spectral_theorem.py                                                                                               
  B =
  [[ 0.37139617 -0.36034904]
   [-0.36034904  2.30840586]]

  V =
  [[-0.85231143  0.36198424  0.37753496]
   [-0.50914829 -0.40898848 -0.75729548]
   [ 0.11972159  0.83767287 -0.53288921]]

  V.T @ V =
  [[ 1.00000000e+00  6.44268216e-17 -6.75467825e-17]
   [ 6.44268216e-17  1.00000000e+00 -1.58338343e-16]
   [-6.75467825e-17 -1.58338343e-16  1.00000000e+00]]

  Lambda (eigenvalues) = [11.95081085  2.37327078  0.30653125]
  L1 error (A, V @ Lambda @ V.T) = 2.6867397195928788e-14

  Compare with np.linalg.eig(A):
  [11.95081085  2.37327078  0.30653125]
  [[-0.85231143 -0.36198424 -0.37753496]
   [-0.50914829  0.40898848  0.75729548]
   [ 0.11972159 -0.83767287  0.53288921]]
  ```

<br>

* **A nondiagonalizable matrix.** The matrix $\bold A = [[1, 0], [1, 1]]$ has eigenvalues are $\lambda_1 = \lambda_2 = 1$ with eigenvectors of the form $\bold v = [0, t]^\top$ for nonzero $t \in \mathbb R$. It follows that $\bold A$ is not diagonalizable since it has at most one linearly independent eigenvectors &mdash; not enough to span $\mathbb R^2.$ <br><br>

* (4.56) **Symmetric product of two symmetric matrices.** Suppose $\bold S$ and $\bold T$ are symmetric matrices. What is the condition so that their product $\bold S \bold T$ is symmetric, i.e. $(\bold S \bold T)^\top = \bold S \bold T$? Observe that $(\bold S \bold T)^\top = \bold T ^\top \bold S ^\top = \bold T \bold S.$ Thus, the product of two symmetric matrices is symmetric if and only if the matrices commute. This works for a very small class of matrices, e.g. zeros or constant diagonal matrices. 
In the case of $2 \times 2$ matrices, this is satisfied most naturally by  matrices with constant diagonal entries &mdash; a quirk that does not generalize to higher dimensions.
The lack of symmetry turns out to be extremely important in machine-learning, multivariate statistics, and signal processing, and is a core part of the reason why linear classifiers are so successful [[Lec 56, Q&A]](https://www.udemy.com/course/linear-algebra-theory-and-implementation/learn/lecture/10738628#questions/13889570/): 
    >  "The lack of symmetry means that $\bold C=\bold B^{-1} \bold A$ is not symmetric, which means that $\bold C$ has non-orthogonal eigenvectors. In stats/data science/ML, most linear classifiers work by using generalized eigendecomposition on two data covariance matrices $\bold B$ and $\bold A$, and the lack of symmetry in $\bold C$ turns a compression problem into a separation problem."

    (???)<br><br>

* (4.57) **Hadamard and standard multiplications are equivalent for diagonal matrices.** This can have consequences in practice. The following code in IPython shows that Hadamard multiplication is 3 times faster than standard multiplication in NumPy.
    ```python
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
    **Remark.** The Frobenius inner product is an inner product on $\mathbb R^{m \times n}$ in the same way that the usual dot product is an inner product on $\mathbb R^{mn}$. It follows that the Frobenius norm $\lVert \cdot \rVert_F$ is a norm as it is induced by the inner product $\langle \cdot, \cdot \rangle_F$ [[Prop. 6]](https://ai.stanford.edu/~gwthomas/notes/norms-inner-products.pdf). As usual for complex matrices we replace the transpose with the conjugate transpose: $\langle \bold A, \bold B\rangle_F =\text{tr}(\bold A^* \bold B)$ and $\lVert \bold A \rVert_F= \sqrt{\text{tr} (\bold A^* \bold A)} = \sqrt{\sum\sum |a_{ij}|^2}.$ These are an inner product and a norm on $\mathbb C^{m \times n}$, as in the real case.  <br><br>


* (4.60) **Other norms.** The **operator norm** is defined as $\lVert \bold A \rVert_p = \sup_{\bold x \neq \bold 0} \lVert \bold A \bold x \rVert_p / \lVert \bold x \rVert_p$ where we use the $p$-norm for vectors with $1 \leq p \leq \infty$. This just measures how much $\bold A$ scales the space, e.g. for isometries $\lVert \bold A \rVert_{p} = 1$. Another matrix norm, which unfortunately bears the same notation, is the **Schatten $p$-norm** defined as $\lVert \bold A  \rVert_p = \left( \sum_{i=1}^r \sigma_i^p \right)^{1/p}$ where $\sigma_1, \ldots, \sigma_r$ are the singular values of $\bold A$. That is, the Schatten $p$-norm is the $p$-norm of the vector of singular values of $\bold A$. Recall that the singular values are the length of the axes of the ellipse, so that Schatten $p$-norm is a cumulative measure of how much $\bold A$ expands the space around it in each dimension.
<br><br>

* (4.60) **Operator norm and singular values.** Note that $\lVert \bold A \rVert_{p} = \sup_{\lVert \bold x \rVert = 1} \lVert \bold A \bold x \rVert_p$ for the operator norm. Recall that the unit circle is transformed $\bold A$ to an ellipse whose axes have length corresponding to the singular values of $\bold A$. Geometrically, we can guess that $\lVert \bold A \rVert_{p} = \max_{j} \sigma_j$  where $\sigma_j$ are the singular values of $\bold A$. Indeed, we can verify this:
    ```python
    In [6]: A = np.random.randn(2, 2)
    In [7]: abs(np.linalg.norm(A, 2) - np.linalg.svd(A)[1][0])
    Out[7]: 0.0
    ```

<br>

* (5.64) **Rank as dimensionality of information.** The rank of $\bold A \in \mathbb R^{m \times n}$ is the maximal number of linearly independent columns of $\bold A.$ It follows that $0 \leq r \leq \min(m, n).$ Matrix rank has several applications, e.g. $\bold A^{-1}$ exists for a square matrix whenever it has maximal rank. In applied settings, rank is used in PCA, Factor Analysis, etc. because rank is used to determine how much nonredundant information is contained in $\bold A.$ <br><br>

* (5.65) **Computing the rank.** How to count the maximal number of linearly independent columns? (1) Row reduction (can be numerically unstable). (2) Best way is to use SVD. The rank of $\bold A$ is the number $r$ of nonzero singular values of $\bold A.$ This is how it's implemented in MATLAB and NumPy. The SVD is also used for rank estimation. Another way is to count the number of nonzero eigenvalues of $\bold A$ provided (!) $\bold A$ has an eigendecomposition. Since this would imply that $\bold A$ is similar to its matrix of eigenvalues. This is in general not true. Instead, we can count the eigenvalues of $\bold A^\top \bold A$ or $\bold A\bold A^\top$ &mdash; whichever is smaller &mdash; since an eigendecomposition for both matrices always exist.

<br>

* (5.65) **Rank can be difficult to calculate numerically.** For instance if we obtain $\sigma_k = 10^{-13}$ numerically, is it a real nonzero singular value, or is it zero? In practice, we set thresholds. The choice of threshold can be arbitrary or domain specific, and in general, introduces its own difficulties. Another issue is noise, adding $\epsilon\bold I$ makes $\bold A = [[1, 1], [1, 1]]$ rank two. <br><br>


* **Rank is well-defined.** That is, dimension is well-defined. Rank of $\bold A$ is defined as the dimension of its column space $\mathsf{C}(\bold A).$ The column space is a well-defined subspace of $\mathbb R^m.$ To get a basis for $\mathsf{C}(\bold A)$ one can proceed iteratively, i.e. if $\bold a_1, \ldots, \bold a_n$ are columns of $\bold A$, then take the largest $j$ such that $\sum_{j=1}^n c_j \bold a_j = \bold 0$ and $c_j \neq 0.$ We can remove this vector $\bold a_j$ such that the remaining columns still span $\mathsf{C}(\bold A).$ Repeat until $r$ columns that is linearly independent, i.e. we cannot anymore perform the iterative step. The remaining columns is a basis for $\mathsf{C}(\bold A)$ by definition. It turns out that while the resulting basis is not unique, the count $r$ is &mdash; so that the dimension is well-defined! Another way to construct a basis is to perform Gaussian elimination along the columns of $\bold A$ as each step preserves the column space. The resulting $r$ pivot columns form a basis for $\mathsf{C}(\bold A).$
<br><br>
To see this, consider any two bases $\bold a_1, \ldots, \bold a_{r_1}$ and $\bar \bold a_1, \ldots, \bar \bold a_{r_2}$ for $\mathsf{C}(\bold A).$ We perform an iteration with the two inductive invariants: (1) length of list is $r_2$ and (2) the list spans $\mathsf{C}(\bold A).$ Append $\bold a_1, \bar \bold a_1, \ldots, \bar \bold a_{r_2}.$ This list is linearly dependent since $\bold a_1 \in \mathsf{C}(\bold A).$ Possibly reindexing, let $\bold a_1$ depend on $\bar \bold a_1$, then $\bold a_1, \bar \bold a_2, \ldots, \bar \bold a_{r_2}$ spans $\mathsf{C}(\bold A).$ Repeat with $\bold a_2, \bold a_1, \bar \bold a_2, \ldots, \bar \bold a_{r_2}.$ Note that $\bold a_2$ cannot depend solely on $\bold a_1$ by linear independence, so that $\bold a_2$ depends on $\bar \bold a_2,$ possibly after reindexing. It follows that $\bold a_2, \bold a_1, \bar \bold a_3, \ldots, \bar \bold a_{r_2}$ spans $\mathsf{C}(\bold A).$ Repeat this until we get all unbarred vectors on the left end of the list. This implies that $r_1 \leq r_2$ necessarily. By symmetry, $r_1 \geq r_2$ so that $r_1 = r_2.$ In other words, all basis for $\mathsf{C}(\bold A)$ have the same length which we define to be its dimension. Note that the rank is the maximal number of linearly independent vectors in the space.
<br><br>
**Remark.** In general, we proved (1) that a spanning set always contains a basis, and (2) that the length of any spanning list of a vector space is always is an upper bound on any linearly independent list of vectors in the vector space.

<br> 

* **Row rank = column rank.** 
  Consider a step in Gaussian elimination along the rows of $\bold A$ resulting in $\tilde \bold A.$ We know that $\mathsf{R}(\bold A) = \mathsf{R}(\tilde\bold A).$ So its clear that the row rank remains unchanged. 
  On the other hand, let's consider the independence of the columns after a step. We know there are $r$ basis vectors in the columns of $\bold A$ and the $n - r$ non-basis vectors are a linear combination of the $r$ basis vectors. 
  WLOG, suppose the first $r$ columns of $\bold A$ form the basis for $\mathsf{C}(\bold A).$ Then, for $j > r$, there exist a vector $\bold x$ such that $\bold A \bold x = \bold 0$ and $x_j = -1$ which encode the dependencies. Moreover, the only solution of the homogeneous system such that $x_j = 0$ for $j > r$ is $\bold x = \bold 0.$ Observe that $\bold A$ and $\tilde\bold A$ have the same null space as an invariant, so that the dependencies in the columns of $\bold A$ and $\tilde \bold A$ are the same. It follows that, the column rank of $\tilde \bold A$ is equal to the column rank of $\bold A.$ Thus, in every step of Gaussian elimination the column and row ranks are invariant. At the end of the algorithm, with $r$ pivots remaining, we can read off that $r$ maximally independent rows and $r$ maximally independent columns. It follows that the column and row ranks of $\bold A$ are equal.

<br>

* **Multiplication with an invertible matrix preserves rank.** This also geometrically makes sense, i.e. automorphism on the input and output spaces. Suppose $\bold U$ is invertible, then $\bold U \bold A \bold x = \bold 0$ if and only if $\bold A \bold x = \bold 0$ so that $\bold U \bold A$ and $\bold A$ have the same null space. Thus, they have the same rank by the rank-nullity theorem. On the other hand, 
  $$
  \text{rank }(\bold A \bold U) = \text{rank } (\bold U^\top \bold A^\top) = \text{rank }\bold (\bold A^\top) = \text{rank }\bold (\bold A)
  $$ 
  since $\bold U^\top$ is invertible and row rank equals column rank. As a corollary, if two matrices $\bold A$ and $\bold B$ are similar, i.e. $\bold A = \bold U \bold B \bold U^{-1}$ for some invertible matrix $\bold U$, then they have the same rank. Another corollary is that the number of nonzero singular values of $\bold A$ is equal to its rank in the decomposition $\bold A = \bold U \bold \Sigma \bold V^\top$ since $\bold U$ and $\bold V^\top$ are both invertible.

<br>

* (5.67) **Generate rank 4 matrix 10x10 matrix randomly by multiplying two randomly generated matrices.** Solution is to multiply 10x4 and 4x10 matrices. Here we assume, reasonably so, that the randomly generated matrices have maximal rank. 

<br>

* (5.69) **Rank of $\bold A^\top \bold A$ and $\bold A \bold A^\top$.** These are all equal to the rank of $\bold A.$ 
The first equality can be proved using by showing the $\mathsf{N} (\bold A^\top \bold A) = \mathsf{N}( \bold A),$ and then invoke the rank-nullity theorem. We used this in the proof of SVD to show conclude that rank $\bold A$ is the number of nonzero singular values of $\bold A.$ 
Now that we know this is true, we can now use the SVD to find the rank of $\bold A \bold A^\top$ which results in $\bold A \bold A^\top = \bold U \bold \Sigma \bold \Sigma^\top \bold U^\top,$ i.e. similar to a diagonal matrix with $r = \text{rank }\bold A$ diagonal entries. 
Thus, $\text{rank } \bold A \bold A^\top = \text{rank }\bold A = r.$ 

<br>

* (5.71) **Making a matrix full-rank by shifting:** $\tilde\bold A = \bold A + \lambda \bold I$ where we assume $\bold A$ is square. This is done for computational stability reasons. Typically the regularization constant $\lambda$ is less than the experimental noise. For instance, if $|\lambda| \gg \max |a_{ij}|,$ then $\tilde \bold A \approx \lambda \bold I$ and $\bold A$ becomes the noise. An exchange in the Q&A highlights another important issue. Hamzah asks:
  > So in a previous video in this section, you talked about how a 3 dimensional matrix spanning a 2 dimensional subspace [...] really is a rank 2 matrix, BUT if you introduce some noise, it can look like like a rank 3 matrix. [...] By adding the identity matrix, aren't you essentially deliberately adding noise to an existing dataset to artificially boost the rank? Am I correct in interpreting that you can possibly identify features in the boosted rank matrix that may not actually exist in the true dataset, and maybe come up with some weird conclusions? If that is the case wouldn't it be very dangerous to increase the rank by adding the identity matrix? Would appreciate some clarification. Thank you!

  To which Mike answers:

  > Excellent observation, and good question. Indeed, there is a huge and decades-old literature about exactly this issue -- how much "noise" to add to data? In statistics and machine learning, adding noise is usually done as part of regularization. <br><br>
  The easy answer is that you want to shift the matrix by as little as possible to avoid changing the data, while still adding enough to make the solutions work. I don't go into a lot of detail about that in this course, but often, somewhere around 1% of the average eigenvalues of the matrix provides a good balance. <br><br>
  Note that this is done for numerical stability reasons, not for theoretical reasons. So the balance is: Do I want to keep my data perfect and get no results, or am I willing to lose a bit of data precision in order to get good results?

<br>

* (5.72) **Is this vector in the span of this set?** Let $\bold x \in \mathbb R^m$ be a test vector. Is $\bold x$ in the span of $\bold a_1, \ldots, \bold a_n \in \mathbb R^m.$ Let $\bold A = [\bold a_1, \ldots, \bold a_n]$ with rank $r.$ The solution is to check whether the rank of $[\bold A | \bold x]$ is equal to the $r$ (in span) or $r+1$ (not in span). <br><br>

* (6.80) **Four Fundamental Subspaces.** Let $\bold A \in \mathbb R^{m \times n}.$ The four fundamental subspaces of $\bold A$ are 
  1. The column space $\mathsf{C}(\bold A) \subset\, \mathbb R^m.$ 
  2. The null space $\mathsf{N}(\bold A) \subset\, \mathbb R^n.$
  3. The row space $\mathsf{C}(\bold A^\top) \subset\, \mathbb R^n.$
  4. The left null space $\mathsf{N}(\bold A^\top) \subset\, \mathbb R^m$, i.e. $\bold A^\top \bold x = \bold 0$ iff. $\bold x^\top \bold A = \bold 0^\top.$

  Recall that column rank equals row rank. In other words, $\dim \mathsf{C}(\bold A) = \dim \mathsf{C}(\bold A^\top).$ We have dual results:

  * $\mathsf{C}(\bold A^\top) \oplus \mathsf{N}(\bold A) =\, \mathbb R^n \;\; \text{s.t.} \;\; \mathsf{C}(\bold A^\top) \perp \mathsf{N}(\bold A);$ and
  * $\mathsf{C}(\bold A) \oplus \mathsf{N}(\bold A^\top) =\, \mathbb R^m \;\; \text{s.t.} \;\; \mathsf{C}(\bold A) \perp \mathsf{N}(\bold A^\top).$ 
  <br><br>

  This implies that $\dim  \mathsf{N}(\bold A^\top) = m-r$ and $\dim {N}(\bold A) = n-r.$ Proving one of the two decompositions proves the other by duality. Suppose $\bold x \in \mathbb R^m$ and consider the SVD $\bold A = \bold U \bold \Sigma \bold V^\top.$ Let $\tilde \bold x = \bold x - \bold U \bold U^\top \bold x.$ Note that $\bold U\bold U^\top \bold x \in \mathsf{C}(\bold A)$ since the columns of $\bold U$ are $\bold u_i = \bold A \bold v_i$ for $i = 1, \ldots, r$ where $r = \text{rank }\bold A,$ and the singular vectors $\bold u_i$ and $\bold v_j$ are, resp., ONBs of $\mathbb R^m$ and $\mathbb R^n.$ Then
  $$
  \bold A^\top \tilde \bold x = \bold V \bold \Sigma \bold U^\top \left( \bold x - \bold U \bold U^\top \bold x \right) = \bold 0.
  $$
  It follows that $\mathbb R^m = \mathsf{N}(\bold A^\top) + \mathsf{C}(\bold A).$ To complete the proof, we show the intersection is zero. Suppose $\bold y \in  \mathsf{N}(\bold A^\top) \cap \mathsf{C}(\bold A).$ Then, $\bold A^\top\bold y = \bold 0$ and $\bold y = \bold A \bold x$ for some $\bold x \in \mathbb R^n.$ Thus, $\bold A^\top \bold A \bold x = \bold 0$ which implies $\bold A \bold x = \bold 0.$ In other words, $\bold y = \bold 0.$ The orthogonality between spaces is straightforward. For instance, $\bold A \bold x = \bold 0$ implies $\bold x \perp \mathsf{C}(\bold A^\top)$; the other follows  by duality. <br><br>

<p align="center">
  <img src="img/fourfundamental.png" alt="drawing" width="500"/>
  </p>
  <br><br>

* **Basis for fundamental subspaces.** Basis for the fundamental subspaces can be obtained from the SVD. We can write $\bold A^\top = \bold V\bold \Sigma \bold U^\top$ so that 
  $$
  \bold A^\top \bold u_i = \sigma_i \bold v_i
  $$ 

  for $i = 1, \ldots, r = \text{rank }\bold A.$ It follows that $\bold v_1, \ldots, \bold v_r$ form a basis for $\mathsf{C}(\bold A^\top),$ while we know that $\bold v_{r+1}, \ldots, \bold v_n$ is a basis for $\mathsf{N}(\bold A).$ Analogously, $\bold u_1, \ldots, \bold u_r$ is a basis for $\mathsf{C}(\bold A)$ while $\bold u_{r+1}, \ldots, \bold u_n$ is a basis for $\mathsf{N}(\bold A^\top).$ Using these ONBs give an easier proof of the orthogonality of subspaces, as well as the decomposition of the input and output spaces. <br><br>

* **Determinant nonzero $\Leftrightarrow$ Full rank.** Consider the SVD of a square matrix $\bold A = \bold U \Sigma \bold V^\top.$ Then 
  $$
  |\det A| = \prod_{i=1}^r \sigma_i.
  $$ 
  Since $\sigma_i > 0$ for $i \leq r$, $\det \bold A \neq 0$ if and only if $r = n.$ Geometrically, this means that the ellipse has zero volume. It can be shown that the determinant is the volume of the image of the unit parallelepiped in the output space. (See Axler.) Consequently, once dimensions have been collapsed the corresponding input vector for each output vector becomes intractable, i.e. $\bold A$ cannot be invertible. (We also have the result that $\bold A^{-1}$ exists iff. $\text{rank }\bold A = n.$) <br><br> 

* (8.98) **Growth of det of shifted random matrix.** In this experiment, we compute the average determinant of $10,000$ shifted $n\times n$ matrices (i.e. we add $\lambda \bold I_n$) with entries $a_{ij} \sim \mathcal{N}(0, 1).$ Moreover, we make two columns dependent so that its determinant is zero prior to shifting. We plot this value as $\lambda$ grows from $0$ to $1$ with $n = 20.$ Explosion: <br><br>

<p align="center">
  <img src="img/8_detgrowth.png" alt="drawing" width="500"/>
  </p>
  <br>

* (10.1) **Full rank $\Leftrightarrow$ Invertible.** Let $r = \text{rank }\bold A.$ This fact is consistent with the SVD, namely, take $\bold A^{-1} = \bold U \bold \Sigma^{-1} \bold V^\top.$ Note that the $\Sigma$ is invertible if and only if $r = n,$ i.e. ${\bold\Sigma^{-1}}_{ii} = \sigma_i^{-1}$ for $i = 1, \ldots, n.$ Without using the SVD, we can prove this using the rank-nullity theorem and the fact that $\bold A$ is invertible (as a matrix) if and only if its corresponding linear operator on $\mathbb R^n$ is invertible (as a map). The corresponding operator for $\bold A$ is invertible by counting dimensions resulting in $\mathsf{C}(\bold A) = \mathbb {R}^n$ (onto) and $\mathsf{N}(\bold A) = \bold 0$ (one-to-one). <br><br>

* (10.1) **Sparse matrix has a dense inverse.** A sparse matrix can have a dense inverse. This can cause memory errors in practice. In `src/10_sparse_to_dense.py` we artificially construct a sparse matrix. This is typically singular, so we shift it to make it nonsingular.  The result is that the inverse is 50x more dense than the original matrix:

  ```
  A sparsity:      0.001999
  A_inv sparsity:  0.108897
  ```

<br>

* (9.108) **Existence of left and right inverses.** Let $\bold A \in \mathbb R^{m\times n}.$ TFAE 
  1. $\text{rank }\bold A = n.$
  2. $\bold A$ is 1-1.
  3. $\bold A$ is left invertible.

  Suppose one of these is true (so that all are true), then we can construct a left inverse using the SVD $\bold A = \bold U \bold \Sigma \bold V^\top$ which allows us to write
  $
  \bold A^\top \bold A = \bold V \bold \Sigma^\top \bold \Sigma \bold V^\top.
  $ 
  This is invertible since $r = n,$ i.e. $\bold\Sigma^\top \bold \Sigma$ is $n \times n$ with nonzero entries on its diagonal. Moreover, the inverse can be efficiently computed using $(\bold A^\top \bold A)^{-1} = \bold V (\bold\Sigma^\top \bold \Sigma)^{-1} \bold V^\top$ where $(\bold\Sigma^\top \bold \Sigma)^{-1} = \bold \Sigma_n^{-2}$ is the diagonal matrix with entries $1/\sigma_j^2$ for $j=1,\ldots, n.$ Then, a left inverse is
    $$
    (\bold A^\top \bold A)^{-1} \bold A^\top.
    $$ 
  Note that we have corresponding dual equivalences about the rows of $\bold A.$ In this case, we have a wide matrix with maximal rows, and a right inverse of $\bold A$ can be constructed as
  $$
  \bold A^\top (\bold A \bold A^\top)^{-1}
  $$
  where $(\bold A \bold A^\top)^{-1} = \bold U (\bold \Sigma \bold \Sigma^\top)^{-1} \bold U^\top$ can be efficiently computed as in the left inverse with $(\bold \Sigma \bold \Sigma^\top)^{-1} = \bold \Sigma_m^{-2}.$

<br>

* (9.111) **Moore-Penrose Pseudo-inverse.** Now that we know how to compute the one sided inverse from rectangular matrices, assuming they have full column rank or full row rank, the big missing piece is what to do with a reduced rank matrix. It turns out that it is possible to find another matrix that is not formally an inverse, but is some kind of a good approximation of what the inverse element should be in a least squares sense (later), i.e. what is called a pseudo-inverse. The **Moore-Penrose pseudo-inverse** for a matrix $\bold A \in \mathbb R^{m \times n}$ is defined as the unique matrix $\bold A^+ \in \mathbb R^{n \times m}$ that satisfies the four Penrose equations:

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
  <br><br>
    ```python
    In [1]: import numpy as np
    
    In [2]: A = np.random.randn(3, 3)
    In [3]: A[:, 0] = A[:, 1] * 3.2 - A[:, 2] * 1.2 # make rank 2
    In [4]: u, s, vT = np.linalg.svd(A)

    In [5]: u[:, :2] @ (u[:, :2].T)
    Out[5]: 
    array([[ 0.89274311,  0.30151708,  0.06957224],
           [ 0.30151708,  0.15238496, -0.19557923],
           [ 0.06957224, -0.19557923,  0.95487192]])

    In [6]: A @ np.linalg.pinv(A)
    Out[6]: 
    array([[ 0.89274311,  0.30151708,  0.06957224],
           [ 0.30151708,  0.15238496, -0.19557923],
           [ 0.06957224, -0.19557923,  0.95487192]])
    ```

<br>

* **Moore-Penrose pseudo-inverse as left and right inverse.** Let $\bold A \in \mathbb R^{m \times n}$ with maximal rank. It turns out the left and right inverses we constructed above is the Moore-Penrose pseudo-inverse of $\bold A$ in each case:

  * $\bold A^+ = (\bold A^\top \bold A)^{-1} \bold A^\top$ (tall)
  
  * $\bold A^+ = \bold A^\top(\bold A \bold A^\top)^{-1}$ (wide) 

  This follows from uniqueness and the fact that the left and right inverses each satisfies the Penrose equations. Any left or right inverse will trivially satisfy the first two equations, but not *both* the third and fourth! For instance, consider `A = np.vstack([ np.eye(3), [0, 0, 1] ])` and a left inverse ` B = np.hstack([np.eye(3), np.array([0, 0, 0]).reshape(-1, 1)])` which is not the Moore-Penrose pseudo-inverse. The product `A @ B` turns out to be not symmetric, as expected.
  
  <br>
  
* **An exercise on consistency.** Recall that $\bold A^+ = \bold V \bold \Sigma^+ \bold U^\top$ uniquely. As an exercise, we want to show that this is consistent with the formula for $\bold A^+$ obtained for matrices with maximal rank. We do this for the tall case $m > n$, the case where the matrix is wide is analogous. Then 
    $$
    \bold A^+ = (\bold A^\top \bold A)^{-1} \bold A^\top
    = \bold V (\bold \Sigma^\top \bold \Sigma)^{-1} \bold \Sigma^\top \bold U^\top.
    $$
    Since $\bold \Sigma$ is a tall matrix with maximal rank as well, we have $\bold \Sigma^+ = (\bold \Sigma^\top \bold \Sigma)^{-1} \bold \Sigma^\top.$ This completes the exercise.

<br>

* 

