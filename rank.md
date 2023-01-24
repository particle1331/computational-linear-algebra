## Rank and dimension

<br>

* 

* 

* 

* 

* 

* 

* 

* (4.8) **Multiplication with an invertible matrix preserves rank.** Suppose $\mathbf U$ is invertible, then $\mathbf U \mathbf A \mathbf x = \mathbf 0$ if and only if $\mathbf A \mathbf x = \mathbf 0$ so that $\mathbf U \mathbf A$ and $\mathbf A$ have the same null space. Thus, they have the same rank by the rank-nullity theorem. On the other hand, 
  $$
  \text{rank }(\mathbf A \mathbf U) = \text{rank } (\mathbf U^\top \mathbf A^\top) = \text{rank }\mathbf (\mathbf A^\top) = \text{rank }\mathbf (\mathbf A)
  $$ 
  since $\mathbf U^\top$ is invertible and row rank equals column rank. As a corollary, if two matrices $\mathbf A$ and $\mathbf B$ are similar, i.e. $\mathbf A = \mathbf U \mathbf B \mathbf U^{-1}$ for some invertible matrix $\mathbf U$, then they have the same rank. Another corollary is that the number of nonzero singular values of $\mathbf A$ is equal to its rank in the decomposition $\mathbf A = \mathbf U \mathbf \Sigma \mathbf V^\top$ since $\mathbf U$ and $\mathbf V^\top$ are both invertible.

  <br>

  **Remark.** This also geometrically makes sense, i.e. automorphism on the input and output spaces. Applying $\mathbf U$ to a basis of $\mathsf C(\mathbf A)$ results in a basis of the same cardinality. So that $\mathsf C(\mathbf U \mathbf A)$ has the same dimension. On the other hand, transforming the input space by $\mathbf U,$ we still get $\mathbb R^n$ so that $\mathsf C(\mathbf A) = \mathsf C (\mathbf A \mathbf U).$ Then, we can prove equality of row and column rank by constructing a CR decomposition by means of left and right multiplying elementary matrices which do not change rank, and whose products have independent columns and rows, respectively. <br><br>

  <p align="center">
  <img src="img/CR_decomp.svg" alt="drawing" width="400"/> <br> 
  <b>Figure.</b> Visualizing the CR decomposition.
  </p>


<br>

* (4.9) **Generate rank 4 matrix 10x10 matrix randomly by multiplying two randomly generated matrices.** Solution is to multiply 10x4 and 4x10 matrices. Here we assume, reasonably so, that the randomly generated matrices have maximal rank. 

<br>

* (4.10) **Rank of $\mathbf A^\top \mathbf A$ and $\mathbf A \mathbf A^\top$.** These are all equal to the rank of $\mathbf A.$ 
The first equality can be proved using by showing the $\mathsf{N} (\mathbf A^\top \mathbf A) = \mathsf{N}( \mathbf A),$ and then invoke the rank-nullity theorem. We used this in the proof of SVD to show conclude that rank $\mathbf A$ is the number of nonzero singular values of $\mathbf A.$ The second equality follows by replacing $\mathbf A$ with $\mathbf A^\top$ and the fact that row rank equals column rank.  
We can also see this from the SVD which gives us $\mathbf A \mathbf A^\top = \mathbf U \mathbf \Sigma \mathbf \Sigma^\top \mathbf U^\top$ i.e. similar to $\Sigma \mathbf \Sigma^\top$ which has $r = \text{rank }\mathbf A$ diagonal entries. 
Thus, $\text{rank } \mathbf A \mathbf A^\top = \text{rank }\mathbf A = r.$ 

<br>

* (4.11) **Making a matrix full-rank by shifting:** $\tilde\mathbf A = \mathbf A + \lambda \mathbf I$ where we assume $\mathbf A$ is square. This is done for computational stability reasons. Typically the regularization constant $\lambda$ is less than the experimental noise. For instance, if $|\lambda| \gg \max |a_{ij}|,$ then $\tilde \mathbf A \approx \lambda \mathbf I$ and $\mathbf A$ becomes the noise. An exchange in the Q&A highlights another important issue. Hamzah asks:
  > So in a previous video in this section, you talked about how a 3 dimensional matrix spanning a 2 dimensional subspace [...] really is a rank 2 matrix, BUT if you introduce some noise, it can look like like a rank 3 matrix. [...] By adding the identity matrix, aren't you essentially deliberately adding noise to an existing dataset to artificially boost the rank? Am I correct in interpreting that you can possibly identify features in the boosted rank matrix that may not actually exist in the true dataset, and maybe come up with some weird conclusions? If that is the case wouldn't it be very dangerous to increase the rank by adding the identity matrix? Would appreciate some clarification. Thank you!

  To which Mike answers:

  > Excellent observation, and good question. Indeed, there is a huge and decades-old literature about exactly this issue -- how much "noise" to add to data? In statistics and machine learning, adding noise is usually done as part of regularization. <br><br>
  The easy answer is that you want to shift the matrix by as little as possible to avoid changing the data, while still adding enough to make the solutions work. I don't go into a lot of detail about that in this course, but often, somewhere around 1% of the average eigenvalues of the matrix provides a good balance. <br><br>
  Note that this is done for numerical stability reasons, not for theoretical reasons. So the balance is: Do I want to keep my data perfect and get no results, or am I willing to lose a bit of data precision in order to get good results?

<br>

* (4.12) **Is this vector in the span of this set?** Let $\mathbf x \in \mathbb R^m$ be a test vector. Is $\mathbf x$ in the span of $\mathbf a_1, \ldots, \mathbf a_n \in \mathbb R^m.$ Let $\mathbf A = [\mathbf a_1, \ldots, \mathbf a_n]$ with rank $r.$ The solution is to check whether the rank of $[\mathbf A | \mathbf x]$ is equal to the $r$ (in span) or $r+1$ (not in span). 