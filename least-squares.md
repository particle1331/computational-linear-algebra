## Least squares for model fitting

<br>

* (9.1) **Linear least squares.** The linear least squares problem is
  $$
  \hat \mathbf w = \argmin_{\mathbf w} \lVert \mathbf X \mathbf w - \mathbf y \rVert^2. 
  $$ 

  Here we use the Euclidean norm. In other words, we want to find the optimal choice of parameters $\mathbf w$ such that gives the best least squares approximation of $\mathbf y$ as a linear combination of columns of $\mathbf X$, i.e. the closest point in $\mathsf{C}(\mathbf X)$ to $\mathbf y.$ 
  
  <br>

  In applications, we use objective is used to model the data as a linear system perhaps under some measurement noise. Here $\mathbf y \in \mathbb R^n$ is a sample of output values, while $\mathbf X \in \mathbb R^{n \times d}$ is a sample of $n$ input values, then $\mathbf w \in \mathbb R^d$ is the weights vector which act as parameters of the model. 

<br>

* (9.2) **Solution to the LLS objective.** 
  Geometrically, it is intuitive that the unique vector in $\mathsf{C}(\mathbf X)$ that minimizes the distance from $\mathbf y$ is the orthogonal projection. Observe that for any $\mathbf z \in \mathsf{C}(\mathbf X)$,
    $$
    \lVert \mathbf z - \mathbf y \rVert^2 = \lVert \mathbf z - \hat\mathbf y \rVert^2 + \lVert \hat\mathbf y - \mathbf y \rVert^2 \geq  \lVert \hat\mathbf y -\mathbf y  \rVert^2.
    $$   

  Thus, projections are solutions to the LLS and we can take $\hat \mathbf w = \mathbf X^+ \mathbf y.$ In fact, projections are the only solutions. We prove this using the singular vectors of $\mathbf X.$ The objective function in terms of the SVD can be written as
    $$
    \begin{aligned}
    \lVert \mathbf y - {\mathbf U \mathbf \Sigma} {\mathbf V}^\top \mathbf w \rVert^2
    &= \lVert {\mathbf U}^\top \mathbf y - {\mathbf \Sigma} {\mathbf V}^\top \mathbf w \rVert^2 \\ 
    &= \lVert {\mathbf U_d}^\top\mathbf y - \mathbf \Sigma_d {\mathbf V_d}^\top \mathbf w \rVert^2 + \lVert {\mathbf U_{d+1:}}^\top\mathbf y \rVert^2. 
    \end{aligned}
    $$

  We ignore the second term since it does not depend on $\mathbf w$ &mdash; this is precisely the normal distance of $\mathbf y$ from $\mathsf{C}(\mathbf X).$ The unique minimal solution is obtained by setting all components of the first term zero, i.e. finding $\mathbf w$ such that ${\mathbf U_d}^\top\mathbf y = \mathbf \Sigma_d {\mathbf V_d}^\top \mathbf w.$ One such solution is
  $$
  \begin{aligned}
  \hat \mathbf w 
    = \sum_{k=1}^r \frac{1}{\sigma_k} \boldsymbol v_k \boldsymbol u_k^\top \mathbf y 
    = \mathbf V \mathbf \Sigma^+ \mathbf U^\top \mathbf y = \mathbf X^+ \mathbf y.
  \end{aligned}
  $$
  
  Note that there is gap of $d - r$ right singular vectors that get zeroed out by $\mathbf \Sigma^+.$ So the most general solution is $\hat \mathbf w = \mathbf X^+ \mathbf y + \sum_{j = r+1}^d \alpha_j \mathbf v_j$ 
  for parameters $\alpha_j \in \mathbb R.$ If the columns are independent, then there is a unique optimal weight. Otherwise, the optimal weights occupy an affine space of $d - r$ dimensions!
  
<br>

* (9.3) **Linear least squares via gradient descent.** Note that the least squares objective can be written as 
  $$
  J(\mathbf w) = \frac{1}{n}\sum_{i=1}^n \left( \sum_{j=1}^d x_{ij} w_j - y_i \right)^2.
  $$

  This is essentially a shallow neural network with identity activation with MSE loss! Then, we can solve this using SGD or batch GD with the gradient step
  $$
  \nabla_k J (\mathbf w) = \frac{2}{n} \sum_{i=1}^n \left( \sum_{j=1}^d x_{ij} w_j - y_i \right) x_{ik}.
  $$

  We will use this to update $\mathbf w = \mathbf{w} - \eta\;\nabla J$ for some fixed learning rate $\eta > 0.$ For each iteration, the weights $\mathbf w$ move to some (locally) optimal weight that minimizes the objective $J.$ For a linear model with nonzero bias term $w_0$, we can set $x_{i0} = 1.$

<br>
  
* (9.4) **Code demo: gradient descent with LLS loss.** In `src/11_leastsquares_descent.py`, we perform gradient descent on a synthetic dataset. For simplicity, i.e. so we can plot, we model the signal $y = -1 + 3 x$ where $x \in [-1, 1]$ and with some Gaussian measurement noise:
  * `X[:, 0] = 1`
  * `X[:, 1] = np.random.uniform(low=-1, high=1, size=n)`
  * `y = X @ w_true + 0.01*np.random.randn(n)` 
  
  where `w_true = np.array([-1, 3])`. The gradient step can be vectorized as follows:
  ```python
  2*((X @ w - y) * X[:, k]).mean()
  ``` 
  for each `k`. Further vectorization requires broadcasting: 
  ```python
  2*((X @ w - y).reshape(-1, 1) * X).mean(axis=0)
  ```
  i.e. multiplies `X @ w - y` to each column of `X` followed by taking the mean of each column. This gives us the gradient vector of length 2. Let us see whether gradient descent can find `w_true`.

  <br>

  <p align="center">
  <img src="img/11_leastsquares_descent.png" title="drawing" />

  </p> 

  <br>

  ```python
  MSE(y, X @ w_true) = 9.343257744523987e-05
  MSE(y, X @ w_best) = 0.01446739159531978
  MSE(y, X @ X_pinv @ y) = 9.32687024471718e-05
  w_true = [-1  3]
  w_best = [-1.00332668  2.79325696]
  X_pinv @ y = [-0.99971352  2.99951481]
  ```

  Here `w_best` is the best weight found using GD. The analytic solution obtained using the pseudoinverse performs better. Try to experiment with the code, e.g. changing the signal to be quadratic (nonlinear) to see how the loss surface will change. It will still be convex, since only the data changes. However, it does not anymore minimize to an MSE proportional equal to the square of the amplitude $a$ of the noise. To derive this, observe that since $\mu = 0$, the variance is $\mathbb E[a^2 X^2] = a^2 \mathbb E[X^2] = a^2\sigma^2 = a^2.$ This agrees with the best MSE of `9.34e-05` ~ `1e-4`. This can be derived analytically by writing the loss function as 
  $(\mathbf y - \mathbf X \mathbf w)^\top (\mathbf y - \mathbf X \mathbf w).$ If we substitute $\mathbf y = \mathbf X \mathbf w_{\text{true}} + \boldsymbol{\epsilon},$ then 
  $$J(\mathbf w) = (\mathbf w - \mathbf w_{\text{true}})^\top \mathbf X^\top \mathbf X (\mathbf w - \mathbf w_{\text{true}}) - 2 \boldsymbol{\epsilon}^\top \mathbf X (\mathbf w - \mathbf w_{\text{true}}) + \boldsymbol{\epsilon}^\top\boldsymbol{\epsilon}. 
  $$

  This is a quadratic surface centered at $\mathbf w_{\text{true}}$ with value $J = \boldsymbol{\epsilon}^\top\boldsymbol{\epsilon}$ at the minimum $\mathbf w = \mathbf w_{\text{true}}.$

<br>

* (9.5) **Loss surfaces.** If $\mathbf X$ has linearly dependent columns, we expect that the optimal weight vector $\mathbf w$ is not unique. The loss surfaces are plotted below, see `11/loss_surface.py`, where we plot the loss surface with $\mathbf X$ having dependent columns (top) with `X[:, 0] = 2 * X[:, 1]` &mdash; observe the whole strip of optimal weights; and the loss surface where $\mathbf X$ has independent columns with a unique optimal point (bottom). Recall that the equation for optimal weights is given by 
  
  $$\hat\mathbf w = \mathbf X^+ \mathbf y + \sum_{j = r+1}^d \alpha_j \mathbf v_j$$ 
    
  for coefficients $\alpha_j \in \mathbb R.$ In this example, $d = 2$ and $r = 1$ so the optimal weights occupy 1-dimension in the parameter space spanned by the second left singular vector $\mathbf v_2$ offset by $\mathbf w^+.$ This is implemented in the code and the optimal weights plotted as a scatterplot. (The 3D plots on the left can be moved around and inspected using `plt.show()` in the script, if you actually run the code!) Note that the optimal points are generated using the equation for the optimal weight (see code), i.e. not manually plotted. Thus, the code demonstrates uniqueness and nonuniqueness of optimal weights depending on the rank of $\mathbf X$ as well as the correctness of the equation. Interesting that the geometry of the loss surface is affected by the rank of $\mathbf X$ and affected in a tractable manner &mdash; i.e. by counting dimensions!

    <br>

    <p align="center">
      <img src="img/11_loss_independent.png"> <br>
      <b>Figure.</b> Loss surface for X with independent columns; unique minimum.
      <br><br>
      <img src="img/11_loss_dependent.png"><br>
      <b>Figure.</b> Loss surface for X with dedependent columns; nonunique (1-dim) minima.
    </p>

<br>