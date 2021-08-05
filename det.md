## Determinant

<br>

* (6.1) **Determinant not zero iff. full rank.** Consider the SVD of a square matrix $\bold A = \bold U \bold \Sigma \bold V^\top.$ Since the determinant of orthogonal matrices is equal to $\pm 1$, then 
  $$
  \left|\det (\bold A) \right| = \prod_{i=1}^n \sigma_i.
  $$ 
  This is nonzero if and only if $r = n,$ i.e. full rank. Geometrically, this means that the ellipse has zero volume. It can be shown that the determinant is the volume of the image of the unit parallelepiped in the output space. Consequently, once dimensions have been collapsed, the corresponding input vector for each output vector becomes intractable, i.e. $\bold A$ cannot be invertible. This is equivalent to $\bold A$ not being full-rank where the column space occupies a surface of lower dimension compared to its output space.

<br>

* (6.2) **Determinant as scale factor for vol. transformation.** To prove the volume formula for the unit parallelepiped, we use the polar decomposition $\bold A = \bold Q \sqrt{\bold A^\top \bold A}$ where $\sqrt{\bold A^\top \bold A} = \bold V \sqrt{\bold \Sigma^\top \bold \Sigma} \bold V^\top$ is a spectral decomposition such that $\bold V$ is an ONB for $\mathbb R^n$ and $\bold Q$ is an isometry, i.e. has determinant $1$ by the product and transpose formula. The unit parallelepiped spanned by $(\bold v_1, \ldots, \bold v_n)$ is transformed to $(\sigma_1 \bold v_1, \ldots, \sigma_n \bold v_n)$ by $\bold A.$ This has (unsigned) volume 
  $$
  \mathsf{vol}(\sigma_1 \bold v_1, \ldots, \sigma_n \bold v_n) = \sigma_1 \ldots, \sigma_n = |\det \bold A \;|.
  $$ 

  The transformation by the isometry $\bold Q$ doesn't change the volume as it does not change distance between points in $\mathbb R^n.$ Thus, $\mathsf{vol}(\bold A \bold v_1, \ldots, \bold A \bold v_n) = |\det \bold A\;|$ where $\bold v_1, \ldots, \bold v_n$ is an ONB for $\mathbb R^n.$
  
<br>

* (6.3) **Growth of det of shifted random matrix.** In this experiment, we compute the average determinant of $10,000$ shifted $n\times n$ matrices (i.e. we add $\lambda \bold I_n$) with entries $a_{ij} \sim \mathcal{N}(0, 1).$ Moreover, we make two columns dependent so that its determinant is zero prior to shifting. We plot this value as $\lambda$ grows from $0$ to $1$ with $n = 20.$ Explosion: <br><br>

    <p align="center">
    <img src="img/8_detgrowth.png" alt="drawing" width="500"/>
    </p>

<br>