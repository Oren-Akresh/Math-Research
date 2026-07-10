import numpy as np

def mod5(A): # takes a matrix and puts all entries modulo 5
    return np.mod(A, 5)

def bracket(A, B): # gives commutator bracket (AB - BA)
    return mod5(A @ B - B @ A)

# GLOBAL VARIABLES (x AND w_i)
x = np.zeros((7,7), dtype=int)
x[2,1] = 1
x[3,2] = 1
x[5,4] = 1
x[6,5] = 1

w = {i : np.zeros((7,7), dtype=int) for i in range(1, 9)}
w[1][0,3] = 1
w[2][0,6] = 1
w[3][1,0] = 1
w[4][4,0] = 1

w[5][1,2] = -1
w[5][2,3] = -1

w[6][1,5] = -1
w[6][2,6] = -1

w[7][4,2] = -1
w[7][5,3] = -1

w[8][4,5] = -1
w[8][5,6] = -1

phi = {}
phi[1] = np.array([[4,0,0,0,0,0,0,0],[0,3,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,2,0,0,0,0],
                    [0,0,0,0,0,0,0,0],[0,0,0,0,0,4,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0]], dtype=int)
phi[2] = np.array([[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,4,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,4,0,0,1],[0,0,0,0,0,4,0,0]], dtype=int)
phi[3] = np.array([[0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,4,0,0,0,0],[0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,4,0],[0,0,0,0,1,0,0,4],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0]], dtype=int)
phi[4] = np.array([[3,0,0,0,0,0,0,0],[0,4,0,0,0,0,0,0],[0,0,2,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,4,0],[0,0,0,0,0,0,0,0]], dtype=int)

def calculate_b(i, j):
    w_i = w[i]
    w_j = w[j]

    first_term = bracket(w_i, bracket(x, bracket(x, w_j)))
    second_term = bracket(bracket(x, w_i), bracket(x, w_j))
    third_term = bracket(bracket(x, bracket(x, w_i)), w_j)

    M_ij = mod5(first_term - second_term + third_term)
    if np.any(bracket(x, M_ij) != 0):
        print(f"M_{i}{j} is not in J_1")
        return None

    adj_matrix = np.zeros((8,8), dtype=int)
    for col_k in range(1, 9):
        basis_expansion = bracket(M_ij, w[col_k])

        for row_m in range(1, 9):
            if row_m in [1, 2, 3, 4]:
                r, c = np.where(w[row_m] != 0)
                val = basis_expansion[r[0], c[0]]
            else:
                r, c = np.where(mod5(w[row_m]) == 4)
                val = mod5(-basis_expansion[r[0], c[0]])
            
            adj_matrix[row_m - 1, col_k - 1] = val
        
    return adj_matrix

def decompose_adj(adj_matrix): # take the adjoint matrix 8x8 and decompose in terms of phi_i
    if adj_matrix is None:
        return None
    
    for c_1 in range(5):
        for c_2 in range(5):
            for c_3 in range(5):
                for c_4 in range(5):
                    guess = mod5(c_1 * phi[1] + c_2 * phi[2] + c_3 * phi[3] + c_4 * phi[4])
                    if np.array_equal(adj_matrix, guess):
                        return (c_1, c_2, c_3, c_4)
                    
for i in range(1, 9):
    for j in range(1, 9):
        b_matrix = calculate_b(i, j)
        coeffs = decompose_adj(b_matrix)

        if coeffs is not None and any(coeffs):
            print(f"b(w_{i}*w_{j}) : ({coeffs[0]}, {coeffs[1]}, {coeffs[2]}, {coeffs[3]})")