import numpy as np
from fractions import Fraction


def inverse(n):
    inverses = {1: 1, 2: 3, 3: 2, 4: 4}
    return inverses[n % 5]

def mod5(A : np.ndarray) -> np.ndarray:  # takes a matrix and puts all entries modulo 5
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

w = {i : np.zeros((8,8), dtype=int) for i in range(1, 6)}

w[1][0,7] = 1
w[1][3,4] = 1

w[2][0,1] = -1
w[2][1,2] = -1
w[2][5,4] = 1
w[2][6,5] = 1

w[3][3,2] = -1
w[3][6,7] = 1

w[4][6,3] = 1
w[4][7,2] = 1

w[5][0,3] = -1
w[5][7,4] = 1




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
y_killed_block = np.zeros((64, 64), dtype=int)
x_killed_block = np.zeros((64, 64), dtype=int)

for idx in range(64):
    r, c = divmod(idx, 8)

    E_rc = np.zeros((8, 8), dtype=int)
    E_rc[r, c] = 1

    symp_result = mod5(np.transpose(E_rc) @ J + J @ E_rc)
    symplectic_block[:, idx] = symp_result.flatten()

    y_killed_result = mod5(bracket(y, E_rc))
    y_killed_block[:, idx] = y_killed_result.flatten()

    x_killed_result = mod5(bracket(x, E_rc))
    x_killed_block[:, idx] = x_killed_result.flatten()


A_system = np.vstack([symplectic_block, y_killed_block, x_killed_block])

nullspace_basis = find_nullspace_mod5(A_system)

j1_heads = [vec.reshape((8, 8)) for vec in nullspace_basis]

P = np.column_stack([w[i].flatten() for i in range(1,6)])
P_trans_P_inv = mod5(np.linalg.inv(mod5(np.transpose(P) @ P)))

# print(P_trans_P_inv)

I = np.eye(8, dtype=int)
phi = []
for head in j1_heads:
    M_64 = mod5(np.kron(head, I) - np.kron(I, np.transpose(head)))
    raw_proj = mod5(np.transpose(P) @ M_64 @ P)
    phi_5 = mod5(P_trans_P_inv @ raw_proj)
    # phi.append(phi_5)

    num_rows = phi_5.shape[0]
    num_cols = phi_5.shape[1]
    phi_copy = np.zeros((num_rows, num_cols), dtype = int)
    for row in range(num_rows):
        for col in range(num_cols):
            val = Fraction(phi_5[row, col])
            numerator = val.numerator
            denominator = val.denominator
            result = (numerator * inverse(denominator)) % 5

            phi_copy[row, col] = result
    phi.append(phi_copy)
for ad in phi:
    print(ad)


# print(1.5 % 5)