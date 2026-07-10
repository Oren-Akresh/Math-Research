import numpy as np

def mod5(A) -> np.ndarray:  # takes a matrix and puts all entries modulo 5
    return np.mod(A, 5)

def bracket(A, B): # gives commutator bracket (AB - BA)
    return mod5(A @ B - B @ A)


x = np.zeros((8, 8), dtype=int) #raising operator

x[1,0] = 1
x[2,1] = 1

x[4,5] = 4
x[5,6] = 4

y = mod5(2 * np.transpose(x))

h = bracket(x, y)

I4 = np.eye(4, dtype=int)
J = np.block([
    [np.zeros((4,4), dtype=int),  I4],
    [-I4,                        np.zeros((4,4), dtype=int)]
])



def inverse(n):
    inverses = {1: 1, 2: 3, 3: 2, 4: 4}
    return inverses[n % 5]

def find_nullspace_mod5(M):
    M = mod5(M.copy())
    num_rows, num_cols = M.shape
    
    pivot_row = 0
    pivots = []

    for col in range(num_cols):
        for r in range(pivot_row, num_rows):
            if M[r, col] != 0:
                M[[pivot_row, r]] = M[[r, pivot_row]]
                break
        else:
            continue

        pivots.append(col)
        inv = inverse(M[pivot_row, col])
        M[pivot_row] = mod5(M[pivot_row] * inv)
        for r in range(num_rows):
            if r != pivot_row and M[r, col] != 0:
                factor = M[r, col]
                M[r] = mod5(M[r] - factor * M[pivot_row])
        pivot_row += 1
        if pivot_row >= num_rows:
            break
    
    free_vars = [c for c in range(num_cols) if c not in pivots]
    basis = []
    for free in free_vars:
        vec = np.zeros(num_cols, dtype=int)
        vec[free] = 1
        for r, pivot_col in enumerate(pivots):
            vec[pivot_col] = mod5(-M[r, free])
        basis.append(vec)
    return basis

symplectic_block = np.zeros((64,64), dtype=int)
weight_block = np.zeros((64, 64), dtype=int)
height_block = np.zeros((64, 64), dtype=int)

for idx in range(64):
    r, c = divmod(idx, 8)

    E_rc = np.zeros((8, 8), dtype=int)
    E_rc[r, c] = 1

    symp_result = mod5(np.transpose(E_rc) @ J + J @ E_rc)
    symplectic_block[:, idx] = symp_result.flatten()

    weight_result = mod5(bracket(h, E_rc) - (3 * E_rc))
    weight_block[:, idx] = weight_result.flatten()

    inner_bracket = bracket(x, E_rc)
    mid_bracket = bracket(x, inner_bracket)
    outer_bracket = bracket(x, mid_bracket)
    height_block[:, idx] = outer_bracket.flatten()

A_system = np.vstack([symplectic_block, weight_block, height_block])

nullspace_basis = find_nullspace_mod5(A_system)

j3_heads = [vec.reshape((8, 8)) for vec in nullspace_basis]
print(len(j3_heads))
