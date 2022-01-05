## Determinant

<br>

* (6.1) **Determinant not zero iff full rank.** Consider the SVD of a square matrix $\mathbf A = \mathbf U \mathbf \Sigma \mathbf V^\top.$ Since the determinant of orthogonal matrices is equal to $\pm 1$, then 
  $$
  \left|\det (\mathbf A) \right| = \prod_{i=1}^n \sigma_i.
  $$ 
  This is nonzero if and only if $r = n,$ i.e. full rank. Geometrically, this means that the ellipse has zero volume. It can be shown that the determinant is the volume of the image of the unit parallelepiped in the output space. Consequently, once dimensions have been collapsed, the corresponding input vector for each output vector becomes intractable, i.e. $\mathbf A$ cannot be invertible. This is equivalent to $\mathbf A$ not being full-rank where the column space occupies a surface of lower dimension compared to its output space.

<br>

* (6.2) **Determinant as scale factor for vol. transformation.** To prove the volume formula for the unit parallelepiped, we use the polar decomposition $\mathbf A = \mathbf Q \sqrt{\mathbf A^\top \mathbf A}$ where $\sqrt{\mathbf A^\top \mathbf A} = \mathbf V \sqrt{\mathbf \Sigma^\top \mathbf \Sigma} \mathbf V^\top$ is a spectral decomposition such that $\mathbf V$ is an ONB for $\mathbb R^n$ and $\mathbf Q$ is an isometry, i.e. has determinant $1$ by the product and transpose formula. The unit parallelepiped spanned by $(\mathbf v_1, \ldots, \mathbf v_n)$ is transformed to $(\sigma_1 \mathbf v_1, \ldots, \sigma_n \mathbf v_n)$ by $\mathbf A.$ This has (unsigned) volume 
  $$
  \mathsf{vol}(\sigma_1 \mathbf v_1, \ldots, \sigma_n \mathbf v_n) = \sigma_1 \ldots, \sigma_n = |\det \mathbf A \;|.
  $$ 

  The transformation by the isometry $\mathbf Q$ doesn't change the volume as it does not change distance between points in $\mathbb R^n.$ Thus, $\mathsf{vol}(\mathbf A \mathbf v_1, \ldots, \mathbf A \mathbf v_n) = |\det \mathbf A\;|$ where $\mathbf v_1, \ldots, \mathbf v_n$ is an ONB for $\mathbb R^n.$
  
<br>

* (6.3) **Growth of det of shifted random matrix.** In this experiment, we compute the average determinant of $10,000$ shifted $n\times n$ matrices (i.e. we add $\lambda \mathbf I_n$) with entries $a_{ij} \sim \mathcal{N}(0, 1).$ Moreover, we make two columns dependent so that its determinant is zero prior to shifting. We plot this value as $\lambda$ grows from $0$ to $1$ with $n = 20.$ Explosion: <br><br>

    <p align="center">
    <img src="img/8_detgrowth.png" alt="drawing" width="500"/>
    </p>

<br>
