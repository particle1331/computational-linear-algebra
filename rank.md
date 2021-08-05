## Rank and dimension

<br>

* (4.1) **Rank as dimensionality of information.** The rank of $\bold A \in \mathbb R^{m \times n}$ is the maximal number of linearly independent columns of $\bold A.$ It follows that $0 \leq r \leq \min(m, n).$ Matrix rank has several applications, e.g. $\bold A^{-1}$ exists for a square matrix whenever it has maximal rank. In applied settings, rank is used in PCA, Factor Analysis, etc. because rank is used to determine how much nonredundant information is contained in $\bold A.$ 

<br>

* (4.2) **Computing the rank.** How to count the maximal number of linearly independent columns? (1) Row reduction (can be numerically unstable). (2) Best way is to use SVD. The rank of $\bold A$ is the number $r$ of nonzero singular values of $\bold A.$ This is how it's implemented in MATLAB and NumPy. The SVD is also used for rank estimation. Another way is to count the number of nonzero eigenvalues of $\bold A$ provided $\bold A$ has an eigendecomposition. Since this would imply that $\bold A$ is similar to its matrix of eigenvalues. This is in general not true. Instead, we can count the eigenvalues of $\bold A^\top \bold A$ or $\bold A\bold A^\top$ &mdash; whichever is smaller &mdash; since an eigendecomposition for both matrices always exist. In practice, counting the rank requires setting a threshold below which values which determine rank are treated as zero, e.g. singular values $\sigma_k$ in the SVD. 

<br>

* (4.3) **Rank can be difficult to calculate numerically.** For instance if we obtain $\sigma_k = 10^{-13}$ numerically, is it a real nonzero singular value, or is it zero? In practice, we set thresholds. The choice of threshold can be arbitrary or domain specific, and in general, introduces its own difficulties. Another issue is noise, adding $\epsilon\bold I$ makes $\bold A = [[1, 1], [1, 1]]$ rank two.

<br>

* (4.4) **Finding a basis for the column space.** We will be particularly interested in finding a subset of the columns of $\bold A$ that is a basis for $\mathsf{C}(\bold A).$ The problem in abstract terms is to find a linearly independent subset of a spanning set that spans the space. One can proceed iteratively. Let $\bold a_1, \ldots, \bold a_n$ be the columns of $\bold A$, take the largest $j$ such that $\sum_{j=1}^n c_j \bold a_j = \bold 0$ and $c_j \neq 0.$ We can remove this vector $\bold a_j$ such that the remaining columns still span $\mathsf{C}(\bold A).$ Repeat until we get $r$ columns that is linearly independent, i.e. $\sum_{j=1}^n c_j \bold a_j = \bold 0$ implies $c_j = 0$ for all $j.$ Further removing any vector fails to span the column space, or the column space is zero in the worst case, so we know this algorithm terminates. <br><br>
Another way to construct a basis for $\mathsf{C}(\bold A)$ is to perform Gaussian elimination along the columns of $\bold A$ as each step preserves the column space. The resulting pivot columns form a basis for $\mathsf{C}(\bold A)$ but is not a subset of the columns of $\bold A.$

<br>

* (4.5) **Basis and dimension.** Basis vectors are linearly independent vectors that span the vector space. We will be interested in finite-dimensional vector spaces, i.e. spaces that are spanned by finitely many vectors. By the above algorithm, we can always reduce the spanning set to a basis, so that a finite-dimensional vector space always has a (finite) basis. We know that a basis is not unique &mdash; in the previous bullet we constructed two bases for $\mathsf{C}(\bold A).$ However, once a basis $\bold v_1, \ldots, \bold v_n$ is fixed, every vector $\bold x$ has a unique representation $(x_1, \ldots, x_n)$ such that $\bold x = \sum_{i=1}^n x_i \bold v_i.$ We can think of this as a **parametrization** of the space by $n$ numbers. It is natural to ask whether there exists a more compressed parametrization, i.e. a basis of shorter length. It turns out that the length of any basis of a finite-dimensional vector space have the same length. Thus, we can think of this number as a property of the space which we define to be its **dimension**. This can be proved with the help of the ff. lemma since a basis is simultaneously spanning and linearly independent:

  > (Finite-dim.) The cardinality of any linearly independent set of vectors is bounded by the cardinality of any spanning set of the vector space.

  <br>

  **Proof.**   The idea for the proof is that we can iteratively exchange vectors in a spanning set with vectors in linearly independent set while preserving the span. 
  Let $\bold u_1, \ldots, \bold u_{s}$ be a linearly independent list and $\bold v_1, \ldots, \bold v_{t}$ be a spanning set in $V.$ We iteratively update the spanning set keeping it at fixed length $t.$ Append $\bold u_1, \bold v_1, \ldots, \bold v_{t}.$ This list is linearly dependent since $\bold u_1 \in V.$ Possibly reindexing, let $\bold u_1$ depend on $\bold v_1$, then $\bold u_1, \bold v_2, \ldots, \bold v_{t}$ spans $V.$ Now append $\bold u_2, \bold u_1, \bold v_2, \ldots, \bold v_{t}.$ Note that $\bold u_2$ cannot depend solely on $\bold u_1$ by linear independence, so that $\bold u_2$ depends on $\bold v_2$ possibly reindexing. That is, we know $c_2 \neq 0$ in the ff. equation:
  $$
  \bold u_2 = c_1 \bold u_1 + \sum_{j=2}^t c_j \bold v_j \implies \bold v_2 = \frac{c_1}{c_2} \bold u_1 - \frac{1}{c_2} \bold u_2 + \sum_{j=3}^t \frac{c_j}{c_2} \bold v_j 
  $$
  so that $\bold u_2, \bold u_1, \bold v_3, \ldots, \bold v_{t}$ spans $V.$ That is, we have exchanged $\bold u_2$ with $\bold v_2$ in the spanning set. Repeat this until we get all $\bold u_j$'s on the left end of the list. This necessarily implies that $s \leq t$ since we cannot run out of $\bold v_i$ vectors in the spanning set due to linear independence of the $\bold u_j$'s. $\square$

<br>

* (4.6) **Obtaining a basis by counting.** This example shows how we can use the bound to reason about linearly independent lists/sets. A linearly independent set of length $r$ where $r$ is the dimension of the space is necessarily a basis. Otherwise, we can append the vector that is not spanned to get a linearly independent list of size $r+1.$

<br>

* (4.7) **Row rank = column rank.** 
  Consider a step in Gaussian elimination along the rows of $\bold A$ resulting in $\tilde \bold A.$ We know that $\mathsf{R}(\bold A) = \mathsf{R}(\tilde\bold A).$ So its clear that the row rank remains unchanged. 
  On the other hand, let's consider the independence of the columns after a step. We know there are $r$ basis vectors in the columns of $\bold A$ and the $n - r$ non-basis vectors are a linear combination of the $r$ basis vectors. 
  WLOG, suppose the first $r$ columns of $\bold A$ form the basis for $\mathsf{C}(\bold A).$ Then, for $j > r$, there exist a vector $\bold x$ such that $\bold A \bold x = \bold 0$ and $x_j = -1$ which encode the dependencies. Moreover, the only solution of the homogeneous system such that $x_j = 0$ for $j > r$ is $\bold x = \bold 0.$ Observe that $\bold A$ and $\tilde\bold A$ have the same null space as an inductive invariant, so that the index of the basis vectors in the columns of $\bold A$ are carried over to $\tilde \bold A.$ It follows that, the column rank of $\tilde \bold A$ is equal to the column rank of $\bold A.$ Thus, in every step of Gaussian elimination the column and row ranks are invariant. At the end of the algorithm, with $r$ pivots remaining, we can read off that $r$ maximally independent rows and $r$ maximally independent columns. It follows that the column and row ranks of $\bold A$ are equal.

<br>

* (4.8) **Multiplication with an invertible matrix preserves rank.** Suppose $\bold U$ is invertible, then $\bold U \bold A \bold x = \bold 0$ if and only if $\bold A \bold x = \bold 0$ so that $\bold U \bold A$ and $\bold A$ have the same null space. Thus, they have the same rank by the rank-nullity theorem. On the other hand, 
  $$
  \text{rank }(\bold A \bold U) = \text{rank } (\bold U^\top \bold A^\top) = \text{rank }\bold (\bold A^\top) = \text{rank }\bold (\bold A)
  $$ 
  since $\bold U^\top$ is invertible and row rank equals column rank. As a corollary, if two matrices $\bold A$ and $\bold B$ are similar, i.e. $\bold A = \bold U \bold B \bold U^{-1}$ for some invertible matrix $\bold U$, then they have the same rank. Another corollary is that the number of nonzero singular values of $\bold A$ is equal to its rank in the decomposition $\bold A = \bold U \bold \Sigma \bold V^\top$ since $\bold U$ and $\bold V^\top$ are both invertible.

  <br>

  **Remark.** This also geometrically makes sense, i.e. automorphism on the input and output spaces. Applying $\bold U$ to a basis of $\mathsf C(\bold A)$ results in a basis of the same cardinality. So that $\mathsf C(\bold U \bold A)$ has the same dimension. On the other hand, transforming the input space by $\bold U,$ we still get $\mathbb R^n$ so that $\mathsf C(\bold A) = \mathsf C (\bold A \bold U).$ Then, we can prove equality of row and column rank by constructing a CR decomposition by means of left and right multiplying elementary matrices which do not change rank, and whose products have independent columns and rows, respectively. <br><br>

  <p align="center">
  <img src="img/CR_decomp.svg" alt="drawing" width="400"/> <br> 
  <b>Figure.</b> Visualizing the CR decomposition.
  </p>


<br>

* (4.9) **Generate rank 4 matrix 10x10 matrix randomly by multiplying two randomly generated matrices.** Solution is to multiply 10x4 and 4x10 matrices. Here we assume, reasonably so, that the randomly generated matrices have maximal rank. 

<br>

* (4.10) **Rank of $\bold A^\top \bold A$ and $\bold A \bold A^\top$.** These are all equal to the rank of $\bold A.$ 
The first equality can be proved using by showing the $\mathsf{N} (\bold A^\top \bold A) = \mathsf{N}( \bold A),$ and then invoke the rank-nullity theorem. We used this in the proof of SVD to show conclude that rank $\bold A$ is the number of nonzero singular values of $\bold A.$ The second equality follows by replacing $\bold A$ with $\bold A^\top$ and the fact that row rank equals column rank.  
We can also see this from the SVD which gives us $\bold A \bold A^\top = \bold U \bold \Sigma \bold \Sigma^\top \bold U^\top$ i.e. similar to $\Sigma \bold \Sigma^\top$ which has $r = \text{rank }\bold A$ diagonal entries. 
Thus, $\text{rank } \bold A \bold A^\top = \text{rank }\bold A = r.$ 

<br>

* (4.11) **Making a matrix full-rank by shifting:** $\tilde\bold A = \bold A + \lambda \bold I$ where we assume $\bold A$ is square. This is done for computational stability reasons. Typically the regularization constant $\lambda$ is less than the experimental noise. For instance, if $|\lambda| \gg \max |a_{ij}|,$ then $\tilde \bold A \approx \lambda \bold I$ and $\bold A$ becomes the noise. An exchange in the Q&A highlights another important issue. Hamzah asks:
  > So in a previous video in this section, you talked about how a 3 dimensional matrix spanning a 2 dimensional subspace [...] really is a rank 2 matrix, BUT if you introduce some noise, it can look like like a rank 3 matrix. [...] By adding the identity matrix, aren't you essentially deliberately adding noise to an existing dataset to artificially boost the rank? Am I correct in interpreting that you can possibly identify features in the boosted rank matrix that may not actually exist in the true dataset, and maybe come up with some weird conclusions? If that is the case wouldn't it be very dangerous to increase the rank by adding the identity matrix? Would appreciate some clarification. Thank you!

  To which Mike answers:

  > Excellent observation, and good question. Indeed, there is a huge and decades-old literature about exactly this issue -- how much "noise" to add to data? In statistics and machine learning, adding noise is usually done as part of regularization. <br><br>
  The easy answer is that you want to shift the matrix by as little as possible to avoid changing the data, while still adding enough to make the solutions work. I don't go into a lot of detail about that in this course, but often, somewhere around 1% of the average eigenvalues of the matrix provides a good balance. <br><br>
  Note that this is done for numerical stability reasons, not for theoretical reasons. So the balance is: Do I want to keep my data perfect and get no results, or am I willing to lose a bit of data precision in order to get good results?

<br>

* (4.12) **Is this vector in the span of this set?** Let $\bold x \in \mathbb R^m$ be a test vector. Is $\bold x$ in the span of $\bold a_1, \ldots, \bold a_n \in \mathbb R^m.$ Let $\bold A = [\bold a_1, \ldots, \bold a_n]$ with rank $r.$ The solution is to check whether the rank of $[\bold A | \bold x]$ is equal to the $r$ (in span) or $r+1$ (not in span). 