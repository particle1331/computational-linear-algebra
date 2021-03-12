# cohen-linalg


My notes and code experiments for linear algebra done SVD. The idea is to construct the SVD as soon as possible, then use it for everything else &mdash; from characterizing invertbility to parametrizing the loss surface of linear regression with linearly dependent data columns. 

* Notes: `notes.md`
* Code: `/src`


The selection and progression of topics follow the course [Complete linear algebra: theory and implementation in code](https://www.udemy.com/course/linear-algebra-theory-and-implementation/) by [Prof. Mike X Cohen](http://mikexcohen.com/), though all content &mdash; and all errors &mdash; in this repo are my own writing.

<!---
<br>

## Syllabus
### 2. Vectors
Algebraic and gemetric interpretations. Vector addition and subtraction. Vector-scalar multiplication. Dot product and its properties. Vector length. Dot product geometry: sign and orthogonality. Vector Hadamard multiplication. Outer product. Cross product. Vectors with complex numbers. Hermitian transpose (a.k.a. conjugate transpose). Interpreting and creating unit vectors. Dimensions and fields in linear algebrra. Subspaces. Span. Linear independence. Basis. 

**Code challenges.** Dot products with matrix columns. Is the dot product commutative? Dot product sign and scalar multiplication. Dot products with unit vectors. <br><br>
  

### 3. Introduction to matrices
Matrix terminology and dimensionality. A zoo of matrices. Matrix addition and subtraction. Matrix-scalar multiplication. Transpose. Complex matrices. Diagonal and trace. Broadcasting matrix arithmetic.

**Code challenges.**
Is matrix-scalar multiplication a linear operation? Linearity of trace. <br><br>


### 4. Matrix multiplications 
Introduction to standard matrix multiplication. Four ways to think about matrix multiplication. Matrix multiplication with a diagonal matrix. Order-of-operations on matrices. Matrix-vector multiplication. 2D transformation matrices. Additive and multiplicative matrix identities. Creating symmetric matrices: additive and multiplicative. Hadamard (element-wise) multiplication. Multiplication of two symmetric matrices. Frobenius dot product. Matrix norms. What about matrix division?

**Code challenges.**
Matrix multiplication by layering (iterating over outer products). Pure and impure rotation matrices. Geometric transformations via matrix multiplication. Symmetry of combined symmetric matrices. Standard and Hadamard multiplication for diagonal matrices. Fourier transform via matrix multiplication! Conditions for self-adjoint.

--->

<br>

## Quick links

* [Proofs involving the Moore-Penrose pseudoinverse](https://en.wikipedia.org/wiki/Proofs_involving_the_Moore%E2%80%93Penrose_inverse)
* [KaTeX Supported Functions](https://katex.org/docs/supported.html)

<br>

## References
* [Sheldon Axler. *Down With Determinants!*. The American Monthly. (1996)](https://www.maa.org/sites/default/files/pdf/awards/Axler-Ford-1996.pdf)
* [Leslie Hogben (editor), *Handbook of Linear Algebra*. CRC Press 2014.](https://www.oreilly.com/library/view/handbook-of-linear/9781466507296/)
* [Cleve Moler. *Numerical Computing with MATLAB*. The MathWorks / SIAM. (2013)](https://www.mathworks.com/moler/index_ncm.html)
* [Peter Olver and Chehzrad Shakiban. *Applied Linear Algebra*. UTM Springer. (2018)](https://www-users.math.umn.edu/~olver/books.html)
* [Petersen & Pedersen. *The Matrix Cookbook*. v. Nov. 15, 2012.](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
