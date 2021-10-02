## Quadratic form and definiteness

<br>

* (11.1) **Quadratic forms.** 
  Let $\bold Q \in \mathbb R^{n\times n}$ be a symmetric matrix. The associated **quadratic form** is defined as $f_\bold Q(\bold x) = \bold x^\top\bold Q \bold x$ for $\bold x \in \mathbb R^n$ to real numbers $\bold x^\top \bold Q \bold x.$ The quadratic form can be interpreted as the corresponding energy function of the matrix. Below we will classify quadratic forms based on its energy profile. And we will see that this profile is intimately connected with the spectrum of the matrix. 
  
  <br>

  **Remark.** If a square matrix $\bold A$ is not symmetric, then we can always symmetrize it in the quadratic form: $\bold x^\top \bold A \bold x =  \bold x^\top\frac{1}{2}\left(\bold A^\top + \bold A\right)\bold x.$ Thus, we can limit the discussion of quadratic forms to symmetric matrices without loss of generality.

<br> 

* (11.2) **Classifying quadratic forms.** A matrix $\bold Q$ is classified according to the possible signs that its quadratic form can take:

  * **Positive definite** if $f_\bold Q(\bold x) > 0$ for nonzero $\bold x.$

  * **Positive semidefinite** if $f_\bold Q(\bold x) \geq 0$ for all $\bold x.$

  * **Indefinite** if $f_\bold Q(\bold x)$ can be negative and positive. 


<br>

<p align='center'>
    <img src='img/quadratic_forms.png'>
    <br>
    <b>Figure.</b> Classification of quadratic forms.
</p>

<br>

* (11.3) **Principal axes theorem and maximal directions.** 
  This is simply an extension of the real spectral theorem. Recall that any real symmetric matrix $\bold Q$ has a spectral decomposition $\bold Q = \bold U \bold \Lambda \bold U^\top$ such that $\bold \Lambda$ is a real matrix of eigenvalues and $\bold U = [\bold u_1, \ldots, \bold u_n]$ is an orthogonal matrix composed of the corresponding orthogonal eigenvectors. This allows us to 'diagonalize' the quadratic form:
  
  $$f_\bold Q (\bold x) = (\bold U^\top \bold x)^\top \bold \Lambda\; (\bold U^\top \bold x).$$

  This makes it clear how the quadratic form acts on an input vector. First, it projects the vector $\bold x$ onto the principal axes getting $x_i = \bold u_i^\top \bold x.$ The resulting vector is dotted to itself weighted by the eigenvalues resulting in $\sum_{i=1}^n \lambda_i {x_i}^2.$ Observe that the principal axes are orthogonal directions of fixed rates of increase or decrease of energy. Assuming $\lambda_1 \geq \ldots \geq \lambda_n,$ then the maximal increase in energy is along $\pm\bold u_1$ where $f_\bold Q(\pm\bold u_1) = \lambda_1.$ On the other hand, the maximal decrease in energy is along $\pm\bold u_n$ where $f_\bold Q(\pm\bold u_n) = \lambda_n.$ For any other direction, we get a suboptimal weighting of eigenvalues.
  
<br>

* (11.4) **Code demo: principal axes of quadratic forms.** In `18_quadratic_form.py`, we verify the theory by plotting the principal axes of each symmetrized matrix in the above figure (except the upper right). The results are shown below. We weigh the eigenvectors with the corresponding eigenvalues which indicates the rate of energy increase along that direction (or decrease if the eigenvalue is negative).

    <br>

    <p align="center">
    <img src="img/18_definiteQF.png"><br>
    <b>Figure.</b> Quadratic form of a definite matrix; unique minimum. <br>
    <br><br>
    <img src="img/18_semidefiniteQF.png"><br>
    <b>Figure.</b> Quadratic form of a definite matrix; nonunique minimum (1-dim). <br>
    <br><br>
    <img src="img/18_indefiniteQF.png"><br>
    <b>Figure.</b> Quadratic form of an indefinite matrix; has a stationary point but no minimum. <br>
    </p>

<br>

* (11.5) **Definiteness and eigenvalues.** 
  Suppose $\bold Q$ has positive eigenvalues, then $f_\bold Q(\bold x) = \sum_{i=1}^n \lambda_i y_i^2 > 0.$ Similarly, $f_\bold Q(\bold x) = \sum_{i=1}^n \lambda_i y_i^2 \geq 0$ whenever $\bold Q$ has nonnegative eigenvalues. Conversely, we can use eigenvector inputs to pick out individual eigenvalues so that positive definiteness implies positive eigenvalues. Similarly, positive semidefiniteness implies having nonnegative eigenvalues. 

  Now suppose $\bold Q$ has eigenvalues of mixed signs. Then, we can pick out these directions to show that $\bold Q$ is indefinite. To prove the converse, suppose $\bold Q$ is indefinite. Let $f_{\bold Q}(\boldsymbol p) > 0$ and let $p_i = \bold u_i^\top \boldsymbol p.$ Then, $\sum_{i=1}^n \lambda_i {p_i}^2 > 0.$ It follows that some $\lambda_i > 0.$ Similarly, assuming $f_{\bold Q}(\boldsymbol q) < 0$ for some $\boldsymbol q$ implies a negative eigenvalue exists. Thus, $\bold Q$ has eigenvalues of mixed signs. 
  
  In summary:

    * All eigenvalues positive iff. $\bold Q$ is positive definite.
    * All eigenvalues nonnegative iff. $\bold Q$ is positive semidefinite.
    * Positive and negative eigenvalues iff. $\bold Q$ is indefinite.

<br>

* (11.6) **Invertibility.** As a consequence of the characterization of the eigenvalues of $\bold Q,$ a positive definite matrix is invertible since it has a trivial null space, whereas a positive semidefinite is noninvertible since it has a nontrivial nullspace which is the eigenspace of zero. 

    <br>

    <p align="center">
    <img src='img/quadratic_form_invertibility.png' width=80%>
    <br>
    <b>Figure.</b> Summary of the results of this section.
    </p>

<br>

* (11.7) **Normalized QF.** One other way of analyzing the energy function is by 'normalizing' it, i.e. computing 
  $$\tilde f_\bold Q(\bold x) = \frac{\bold x^\top \bold Q \bold x}{\bold x^\top\bold x} = \sum_{i,j=1}^n q_{ij} \frac{x_i x_j}{\sum_{i=1}^n {x_i}^2}.$$

  Since we are only dividing the the norm squared of $\bold x,$ the two principal axes $\pm\bold u_1$ and $\pm\bold u_n$ should still be the same directions of greatest increase and decrease. Indeed, this is verified by the ff. plots generated in `18_normalized_QF.py`. Observe that, unlike before, the plots are now bounded, i.e. by $\lambda_1 = \sup \tilde f_\bold Q$ and $\lambda_n = \inf \tilde f_\bold Q.$ It has, however, a singularity at the origin. Having 'fixed rate' of energy increase along the principal axes, it is now more straightforward to see that the energy function actually fixed along these directions: $\tilde f_\bold Q(a\bold u_i) = \lambda_i$ for any scalar $a.$ 

    <br>

    <p align="center">
    <img src="img/18_normalized_definiteQF.png"><br>
    <b>Figure.</b> Energy surface of a definite matrix. <br>
    <br><br>
    <img src="img/18_normalized_semidefiniteQF.png"><br>
    <b>Figure.</b> Energy surface of a semidefinite matrix. <br>
    <br><br>
    <img src="img/18_normalized_indefiniteQF.png"><br>
    <b>Figure.</b> Energy surface of an indefinite matrix.  <br>
    </p>


<br>