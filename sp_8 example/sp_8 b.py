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

phi = {}
phi[1] = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0]
], dtype=int)

phi[2] = np.array([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=int)

phi[3] = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0]
], dtype=int)

phi[4] = np.array([
    [4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 4]
], dtype=int)

phi[5] = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0],
    [1, 0, 0, 0, 0]
], dtype=int)

phi[6] = np.array([
    [4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=int)

P = np.column_stack([w[i].flatten() for i in range(1,6)])
P_trans_P_inv = mod5(np.linalg.inv(mod5(np.transpose(P) @ P)))

def calculate_b(i, j):
    w_i = w[i]
    w_j = w[j]

    first_term = bracket(w_i, bracket(x, bracket(x, w_j)))
    second_term = bracket(bracket(x, w_i), bracket(x, w_j))
    third_term = bracket(bracket(x, bracket(x, w_i)), w_j)

    M_ij = mod5(first_term - second_term + third_term)
    if np.any(bracket(x, M_ij) != 0):
        print(f"M_{i}{j} not in J_1")
        return None
    
    I = np.eye(8, dtype=int)
    M_64 = mod5(np.kron(M_ij, I) - np.kron(I, np.transpose(M_ij)))
    raw_proj = mod5(np.transpose(P) @ M_64 @ P)
    phi_5 = mod5(P_trans_P_inv @ raw_proj)

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
    
    return phi_copy

def decompose_adj(adj_matrix): # i know it's brute forcing this, but it works now for small cases
    if adj_matrix is None:
        return None
    
    for c_1 in range(5):
        for c_2 in range(5):
            for c_3 in range(5):
                for c_4 in range(5):
                    for c_5 in range(5):
                        for c_6 in range(5):
                            guess = mod5(c_1 * phi[1] + c_2 * phi[2] + c_3 * phi[3] + c_4 * phi[4] + c_5 * phi[5] + c_6 * phi[6])
                            if np.array_equal(adj_matrix, guess):
                                return (c_1, c_2, c_3, c_4, c_5, c_6)

for i in range(1, 6):
    for j in range(1, 6):
        b_matrix = calculate_b(i, j)
        coeffs = decompose_adj(b_matrix)

        if coeffs is not None and any(coeffs):
            print(f"b(w_{i}*w_{j}) = ({coeffs[0]}, {coeffs[1]}, {coeffs[2]}, {coeffs[3]}, {coeffs[4]}, {coeffs[5]})")
