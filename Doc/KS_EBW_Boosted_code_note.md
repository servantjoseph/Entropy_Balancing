

## np.linalg.lstsq(H, g) in eb_fit
"Even if H is singular, lstsq can often provide a solution that minimizes the Euclidean 2-norm ||H * step - g||":
This statement refers to the fundamental difference between np.linalg.solve and np.linalg.lstsq when dealing with systems where the matrix H (often denoted A in general linear algebra) is singular or rectangular.

np.linalg.solve(H, g) is designed for systems where H is a square, full-rank (non-singular) matrix. It finds the exact unique solution step such that H * step = g.

np.linalg.lstsq(H, g), on the other hand, is a more general function. When H is singular or even rectangular (more equations than unknowns, or vice versa), an exact solution H * step = g might not exist, or there might be infinitely many solutions. In such cases, np.linalg.lstsq doesn't look for an exact solution. Instead, it finds a step vector that makes H * step as close as possible to g in the sense of the Euclidean 2-norm. This means it minimizes the quantity ||H * step - g||.

Euclidean 2-norm (||vector||): This is the standard length of a vector, calculated as the square root of the sum of the squares of its components. So, ||H * step - g|| represents the 'distance' between the vector H * step (the result of applying the transformation H to step) and the target vector g.

Minimization: By minimizing this norm, lstsq provides the 'best fit' solution in a least-squares sense, even if H is singular. It effectively projects g onto the column space of H and finds the step that achieves this projection. This makes it a powerful fallback mechanism when np.linalg.solve fails, ensuring that the optimization process can continue with a reasonable step even under challenging numerical conditions.

__np.linalg.lstsq__ solves the linear least squares problem by finding a vector $\mathbf{x}$ (the step vector) that minimizes the squared Euclidean 2-norm of the residual vector $\mathbf{r} = H\mathbf{x} - \mathbf{g}$. Mathematically, it minimizes the objective function 
$E(\mathbf{x}) = \Vert{}H\mathbf{x} - \mathbf{g}\Vert{}_2^2 = (H\mathbf{x} - \mathbf{g})^T(H\mathbf{x} - \mathbf{g})$. 

Mathematical Formulation 

• The Cost Function: $\min_{\mathbf{x}} \Vert{}H\mathbf{x} - \mathbf{g}\Vert{}_2^2$, which is $\sum{i=1}^{m} (h_i^T \mathbf{x} - g_i)^2$ 

• Normal Equations (Full Rank Case):When $H$ has full column rank, the minimum is found by setting the gradient of the cost function to zero, leading to the normal equations: $H^T H \mathbf{x} = H^T \mathbf{g} \implies \mathbf{x} = (H^T H)^{-1} H^T \mathbf{g}$ 

• General Solution via Moore-Penrose Pseudoinverse:When $H$ is singular or rectangular, the matrix $H^T H$ is not invertible. NumPy uses the Singular Value Decomposition (SVD) of $H = U \Sigma V^T$ to compute the unique minimum-norm solution via the pseudoinverse $H^\dagger$: $\mathbf{x} = H^\dagger \mathbf{g}$ 




