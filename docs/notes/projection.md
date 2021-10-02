## Projection and orthogonalization

<br>

* (8.1) **Orthogonal projection: definition and uniqueness.** 
  The projection of $\bold y$ onto $\mathsf{C}(\bold A)$ is the unique vector $\hat\bold y$ such that (1) $\hat\bold y \in \mathsf{C}(\bold A)$, and (2) $(\bold y - \hat\bold y) \perp \mathsf{C}(\bold A).$ To show uniqueness, suppose $\hat\bold y_1$ and $\hat\bold y_2$ are two orthogonal vectors to $\bold y.$ Then,
    $$\lVert\bold y - \hat\bold y_2 \rVert^2 = \lVert\bold y - \hat\bold y_1 \rVert^2 + \lVert\hat\bold y_1 - \hat\bold y_2 \rVert^2 \geq \lVert\bold y - \hat\bold y_1 \rVert^2.$$
  
  By symmetry, $\lVert\bold y - \hat\bold y_1 \rVert^2  = \lVert\bold y - \hat\bold y_2 \rVert^2.$ Thus, $\lVert\hat\bold y_2 - \hat\bold y_1 \rVert^2 = 0$ which implies $\hat\bold y_1 = \hat\bold y_2.$ Now that we have shown uniqueness, we proceed with a constructive proof of its existence.

<br>  

* (8.2) **Orthogonal projection: independent columns.** Suppose $\bold A \in \mathbb R^{m \times n}$ has linearly independent columns and $\bold y$ be any vector on the output space $\mathbb R^m.$ To find the projection of $\bold y$ in $\mathsf{C}(\bold A),$ we solve for weights $\bold x$ such that $\bold A^\top( \bold y - \bold A \bold x ) = \bold 0$ getting $\bold x = (\bold A^\top \bold A)^{-1} \bold A^\top \bold y = \bold A^+ \bold y.$ Thus, $\hat\bold y = \bold A \bold A^+ \bold y$ which allows us to define the projection operator onto $\mathsf{C}(\bold A)$ as
  $$
  \begin{aligned}
  P_{\bold A} 
  = \bold A (\bold A^\top \bold A)^{-1} \bold A^\top = \bold A \bold A^+.
  \end{aligned}
  $$
  
<br>

* (8.3) **Orthogonal projection: general case.** Does $P_{\bold A} = \bold A \bold A^+$ hold in the general case? Recall that the right singular vectors $\boldsymbol u_1, \ldots, \boldsymbol u_r$ form a basis for $\mathsf{C}(\bold A).$ It follows that we can decompose $\bold y$ into two components, one orthogonal and one parallel to the subspace:
    $$
    \bold y = \left(\sum_{i=1}^r \boldsymbol u_i \boldsymbol u_i^\top \bold y\right) + 
    \left( \sum_{i=r+1}^{m} \boldsymbol u_i \boldsymbol u_i^\top \bold y \right).
    $$
    
    Then, the orthogonal projection of $\bold y$ can be constructed as $\hat\bold y = \sum_{i=1}^r \boldsymbol u_i \boldsymbol u_i^\top \bold y.$ This is clear from the fact that $(\bold y - \hat\bold y) \perp \mathsf{C}(\bold A)$ and $\hat\bold y \in \mathsf{C}(\bold A),$ and the uniqueness of such a vector. We now prove the claim that $\hat\bold y = \bold A \bold A^+ \bold y.$ This is actually pretty trivial:
    $$
    \bold A \bold A^+ = {\bold U \bold \Sigma \bold \Sigma}^+ \bold U^\top = \sum_{i=1}^r \boldsymbol u_i \boldsymbol u_i^\top.
    $$
    
    Note that unlike the previous case where the columns of $\bold A$ are independent, the weights that make up the projection vector is not anymore unique. Discussion of nonuniqueness of weights is continued in the next bullet.

    <br>

    ```python
    In [39]: A = np.random.randn(3, 3)
    In [40]: y = np.array([1, 1, 2]).reshape(-1, 1)
    In [41]: A[:, 2] = 2 * A[:, 1]
    In [42]: A @ np.linalg.pinv(A) @ y
    Out[42]: 
    array([[0.91468654],
           [0.11846404],
           [2.08741224]])

    In [43]: A[:, [0, 1]] @ np.linalg.pinv(A[:, [0, 1]]) @ y
    Out[43]: 
    array([[0.91468654],
           [0.11846404],
           [2.08741224]])
    ```

<br>

* (8.4) **Moore-Penrose pseudoinverse as left inverse: a wider perspective.** 
  Interestingly, the  orthogonal projection involves the Moore-Penrose pseudoinverse $\bold A^+$ which is a left inverse for $\bold A$ when the columns of $\bold A$ are independent. 
  This can actually be read off from $\bold A^+ \bold y = \bold V_r \bold \Sigma^+_r \bold U_r^\top \bold y.$ Note that $\bold U \bold U^\top \bold y = \bold y$ and $\bold U_r^\top \bold y$ is the components of the projection of $\bold y$ onto $\mathsf{C}(\bold A)$ with respect to the right singular vectors. Since the pseudoinverse $\bold \Sigma^+$ pads latter columns and rows with zero, we only get to invert that part of the vector that is in the column space of $\bold A,$ meanwhile the part that is normal to $\mathsf{C}(\bold A)$ is zeroed out. This is essentially what $\bold A \bold A^+ = \bold U \bold \Sigma \bold \Sigma^+ \bold U^\top = \sum_{i=1}^r \boldsymbol u_i \boldsymbol u_i^\top$ tells us. If $\bold y \in \mathsf{C}(\bold A)$, then $\bold A^+ \bold y$ gives a left inverse of $\bold y.$ The bigger picture is that the pseudoinverse gives the weights to reconstruct the projection of $\bold y$ which, in this case, is itself since it lies in $\mathsf{C}(\bold A).$ 
  
<br>

* (8.5) **First Penrose equation.** 
  If $\bold A$ does not have independent columns, then $\bold A^+ \bold A \neq \bold I.$ This is a consequence of the non-uniqueness of the weights that reconstructs the projection. Suppose $\bold y \in \mathsf{C}(\bold A),$ then $\bold A \bold A^+ \bold A = \bold A$ even if $\bold A^+ \bold A \neq \bold I$ (from the axioms). That is, we can get $\bold A^+ (\bold A \bold w_1) = \bold w_2$ where $\bold w_1 \neq \bold w_2$ and $\bold A \bold w_1 = \bold A \bold w_2.$ This is exactly what this equation means. The second equation is the same but for right invertibility.

<br>

* (8.6) **Projection matrix properties.** (1) ${P_{\bold A}}^2 = P_{\bold A}$ so it reduces to the identity when restricted to $\mathsf{C}(\bold A)$ and (2) ${P_{\bold A}}^\top = P_{\bold A}$ the projection matrix is orthogonal. The eigenvalues of projection matrices are either zero or one as a consequence of (1).
  In the special case of projecting onto a 1-dimensional subspace of $\mathbb R^2$ spanned by the vector $\boldsymbol a,$ we get
    $$
    \begin{aligned}
    P_{\bold A} \bold y 
    &= \boldsymbol a (\boldsymbol a^\top \boldsymbol a)^{-1} \boldsymbol a^\top \bold y \\
    &= \boldsymbol a \lVert \boldsymbol a \rVert^{-2} \lVert \boldsymbol a \rVert \lVert \bold y \rVert \cos \theta \\
    &= \lVert \bold y \rVert \cos \theta\; \hat \boldsymbol a.
    \end{aligned}
    $$


<br>

* (8.7) **Code demo:** `src/10_projection.py`. We confirm computationally that $P_{\bold A} \bold y \perp (\bold y - P_{\bold A} \bold y)$ and plot the resulting vectors. Algebraically, this is equivalent to ${P_{\bold A}}^\top (\bold I - P_{\bold A}).$ 
  
  <br>

  <p align="center">
      <img src="img/10_projection.png" title="drawing" width=60% />
  </p> 

  <br>

  ```python
  (Ax - b) @ Ax = -2.3678975447083417e-16
  ```

<br>

* (8.8) **Projection matrix with orthonormal columns.** Suppose $\bold U$ be an $m \times n$ matrix with columns $\boldsymbol u_1, \ldots, \boldsymbol u_n$ in $\mathbb R^m$ that are orthonormal in $\mathbb R^m.$ Then, $\bold U^\top \bold U = \bold I_n$ so that $\bold U^+$ reduces to $\bold U^\top$. Thus
  $$
  \boxed{P_{\bold U} = \bold U \bold U^\top = \sum_{i=1}^n \boldsymbol u_j \boldsymbol u_j^\top.}
  $$

  This makes sense, i.e. we simply project into each unit vector. Since the vectors are orthonormal, there will be no redundancy in the projection. The job of the factor $(\bold A^\top \bold A)^{-1}$ in the general formula is to correct this redundancy.

<br>

* (8.9) **Gram-Schmidt process.** Given the columns of $\bold A,$ we want to construct an orthonormal basis for $\mathsf{C}(\bold A).$ To do this, we can perform what is called the Gram-Schmidt process. Let $\boldsymbol a_1, \ldots, \boldsymbol a_n$ be the columns of $\bold A.$ Then an ONB $\boldsymbol u_1, \ldots, \boldsymbol u_r$ for $\mathsf{C}(\bold A)$ can be constructed as follows:
  1. $\boldsymbol u_1 = \dfrac{\boldsymbol a_1}{\lVert \boldsymbol a_1 \rVert}.$
  2. $\boldsymbol u_k =  \dfrac{{\boldsymbol a_k - \sum_{j=1}^{k-1} \boldsymbol u_{j} \boldsymbol u_{j}^\top \boldsymbol a_k}}{\lVert {\boldsymbol a_k - \sum_{j=1}^{k-1} \boldsymbol u_{j} \boldsymbol u_{j}^\top \boldsymbol a_k} \rVert} = \dfrac{\boldsymbol a_k - \bold U_{k-1} \bold U_{k-1}^\top \boldsymbol a_k}{\lVert {\boldsymbol a_k - \bold U_{k-1} \bold U_{k-1}^\top \boldsymbol a_k}\rVert}.$ 
  
  where $\bold U_{k-1} = [\boldsymbol u_1 | \ldots | \boldsymbol u_{k-1}].$ That is we remove the component of $\boldsymbol a_k$ projected in the space already spanned by the earlier vectors. The resulting vector is $\boldsymbol u_k$ orthogonal to $\mathsf{C}(\bold U_{k-1}).$

<br>

* (8.10) **Modified Gram-Schmidt.** We introduce a more numerically stable version of Gram-Schmidt which corrects intermediate errors when projecting. Observe that in the Gram-Schmidt process described above, the vector is projected in the whole space $\mathsf{C}(\bold U_{k-1}).$ In the modified version, at step $k$, we remove all components of later vectors that is in the span of $\boldsymbol a_k.$ 
  1. Copy $\boldsymbol v_k = \boldsymbol a_k$ for $k = 1, \ldots, n.$
  2. Normalize $\boldsymbol u_k = \boldsymbol v_k / \lVert \boldsymbol v_k \rVert,$ then update $\boldsymbol v_j = \boldsymbol v_j -  \boldsymbol u_k \boldsymbol u_k^\top \boldsymbol v_j$ for $j > k.$ 
  
  The modification is that instead of projecting the column vector on the whole subspace spanned by earlier vectors, each vector is iteratively projected in the 1-dimensional subspace spanned by earlier vectors. In exact arithmetic, this algorithm returns the same set of orthonormal vectors as the classical GS (use pen and paper to calculate three vectors, i.e. proof by $n=3$). However, the modified GS is more numerically stable as we will show experimentally. Perhaps one reason is that errors are projected away in each prior iteration.

<br>

* (8.11) **Code demo: stability of GS algorithms**. In `src/10_stability_gram-schmidt.py`, we implement the two algorithms and apply it a matrix that almost has identical columns, i.e. the matrix 
  $$ \bold A = 
  \begin{bmatrix}
    1 & 1 & 1 \\
    \epsilon & 0 & 0 \\
    0 & \epsilon & 0 \\
    0 & 0 & \epsilon
  \end{bmatrix}.
  $$
  where $\epsilon = 10^{-8}.$ We compute how close the results is to being orthonormal, i.e. calculate the L1 error $\lVert \bold U^\top \bold U - \bold I_m \rVert_1$:

  ```python
  In [78]: %run 10_stability_gram-schmidt.py
  L1 error (classical GS) = 0.010203694116029399
  L1 error (modified GS) = 1.5250564655067275e-10
  ```

<br>

* (8.12) **QR decomposition.** 
    We can write $\bold A = \bold Q \bold R$ where $\bold Q$ is an $m \times m$
    orthogonal matrix obtained by extending the Gram-Schmidt basis to an ONB of $\mathbb R^m,$ and 
    $\bold R = \bold Q^\top \bold A.$ 
    Note that the entries of $\bold R$ are $r_{ij} = \boldsymbol q_i^\top \boldsymbol a_j.$ But $\boldsymbol q_j = \gamma (\boldsymbol a_j - \bold {Q}_{j-1} \bold Q_{j-1}^\top \boldsymbol a_j)$ for some scalar $\gamma.$ Thus,
    $$
    \gamma^{-1}\boldsymbol q_j + \bold {Q}_{j-1} \bold Q_{j-1}^\top \boldsymbol a_j=  \boldsymbol a_j.
    $$
    This means $\boldsymbol a_j \in \mathsf{C}(\bold Q_{j}).$ But for $i > j$, by construction, $\boldsymbol q_i \perp \mathsf{C}(\bold Q_j)$ which implies $r_{ij} = {\boldsymbol q_i}^\top \boldsymbol a_j = 0.$ The idea is that later Gram-Schmidt vectors are orthogonal to earlier column vectors &mdash; which are spanned by earlier GS vectors. It follows that $\bold R$ is upper triangular. 

<br>

* (8.13) **Computing the Gram-Schmidt in Numpy.** To perfom the Gram-Schmidt algorithm on the columns of a matrix `A` in numpy, simply call `Q, R = np.linalg.qr(A)` to get the orthogonal matrix `Q` having the same colum span as `A`. 
  
  ```python
  >>> A = np.random.randn(20, 20)
  >>> Q, R = np.linalg.qr(A)
  >>> np.abs(Q @ Q.T - np.eye(20)).mean()
  7.281778314245426e-17
  >>> np.abs(Q.T @ Q - np.eye(20)).mean()
  6.498340689575483e-17
  ```

<br>

* (8.14) **Inverse from QR.** The QR decomposition allows for easy computation of the inverse: 
  $$
  \boxed{\phantom{\Big]}\bold A^{-1} = \bold R^{-1} \bold Q^\top.\phantom{\Big]}}
  $$ 

  The inverse of $\bold R$ is faster to compute since it is upper triangular. An experiment for this is done in `src/10_solve_triangular.py` with the ff. results:

  <br>

  <p align="center">
  <img src="img/10_solve_triangular.png" title="drawing"/>

  <b>Figure.</b> Wall time for computing the inverse of a full (blue) and upper triangular (orange) randomly generated n-by-n matrix. 
  </p> 

<br>

* (8.15) **Sherman-Morrison inverse.** From [(24)](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf), $\det( \bold I + \boldsymbol u \boldsymbol v^\top) = 1 + \boldsymbol v^\top \boldsymbol u.$ Thus, the identity perturbed by a rank $1$ matrix is invertible if and only if $1 + \boldsymbol v^\top \boldsymbol u \neq 0.$ In this case the we have a formula for the inverse:
  $$
  \boxed{\left(\bold I + \boldsymbol u \boldsymbol v^\top\right)^{-1} = \bold I - \dfrac{\boldsymbol u \boldsymbol v^\top}{1 + \boldsymbol v^\top \boldsymbol u}.}
  $$
  
<br>