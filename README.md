# Computational Linear Algebra

**Note:** You might need to install an extension such as [Github Math Dislay](https://chrome.google.com/webstore/detail/github-math-display/cgolaobglebjonjiblcjagnpmdmlgmda) to display mathematical formulae.

<br>

Notes and code experiments for linear algebra in code. The idea is to construct the SVD as soon as possible, then use it for everything else &mdash; from characterizing invertbility, to parametrizing the loss surface of a linear regression model. Some of the interesting stuff that are covered:
  * Proof of the real spectral theorem, and a code demo
  * Proof of the singular value decomposition (SVD)
  * An extensive discussion of the Moore-Penrose pseudoinverse
  * Stability of the Gram-Schmidt algorithm
  * Characterizing the loss surface of a linear regression problem
  * Characterizing quadratic forms using the principal axes theorem.

<br>

<p align="center">
    <img src='img/13_kde.png'>
    <br>
    <b>Figure.</b> SVD of a sum of Gaussians. Only the first few vectors are meaningful, the rest model noise. 
</p>

<br>

<p align="center">
    <img src='img/18_normalized_indefiniteQF.png'>
    <br>
    <b>Figure.</b> Energy surface of an indefinite matrix. It has a negative minimum and a positive maximum.
</p>

<br>

## Contents

1. [Vectors and matrices](https://github.com/particle1331/computational-linear-algebra/blob/master/vectors-and-matrices.md)
2. [Singular value decomposition](https://github.com/particle1331/computational-linear-algebra/blob/master/svd.md)
3. [Matrix multiplication and norms](https://github.com/particle1331/computational-linear-algebra/blob/master/mm-norms.md)
4. [Rank and dimension](https://github.com/particle1331/computational-linear-algebra/blob/master/rank.md)
5. [Four fundamental subspaces](https://github.com/particle1331/computational-linear-algebra/blob/master/four-subspaces.md)
6. [Determinant](https://github.com/particle1331/computational-linear-algebra/blob/master/det.md)
7. [Matrix inverse and pseudoinverse](https://github.com/particle1331/computational-linear-algebra/blob/master/inverse.md)
8. [Projection and orthogonalization](https://github.com/particle1331/computational-linear-algebra/blob/master/projection.md)
9. [Least squares for model fitting](https://github.com/particle1331/computational-linear-algebra/blob/master/least-squares.md)
10. [Eigendecomposition](https://github.com/particle1331/computational-linear-algebra/blob/master/eigendecomp.md)
11. [Quadratic form and definiteness](https://github.com/particle1331/computational-linear-algebra/blob/master/quadratic.md)

<br>

## Quick links

* [Proofs involving the Moore-Penrose pseudoinverse](https://en.wikipedia.org/wiki/Proofs_involving_the_Moore%E2%80%93Penrose_inverse)
* [KaTeX Supported Functions](https://katex.org/docs/supported.html)


<br>

## References
* [Mike X Cohen.](http://mikexcohen.com/) [*Complete linear algebra: theory and implementation in code*. Udemy. (2021)](https://www.udemy.com/course/linear-algebra-theory-and-implementation/)
* [Sheldon Axler. *Down With Determinants!* The American Monthly. (1996)](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf)
* [Leslie Hogben (editor). *Handbook of Linear Algebra*. CRC Press. (2014)](https://www.oreilly.com/library/view/handbook-of-linear/9781466507296/)
* [Cleve Moler. *Numerical Computing with MATLAB*. The MathWorks / SIAM. (2013)](https://www.mathworks.com/moler/index_ncm.html)
* [Peter Olver and Chehzrad Shakiban. *Applied Linear Algebra*. UTM Springer. (2018)](https://www-users.math.umn.edu/~olver/books.html)
* [Petersen & Pedersen. *The Matrix Cookbook*. v. Nov. 15, 2012. (2012)](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
