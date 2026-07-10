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

for i in range(1, 6):
    for j in range(1, 6):
        x_wi = bracket(x, w[i])
        x_wj = bracket(x, w[j])

        matrix = mod5(bracket(x_wi, w[j]) - bracket(w[i], x_wj))

        if np.any(matrix != 0):
            coeffs = [0] * 5
            remainder = np.copy(matrix)

            for k in range(1, 6):
                nonzero_indices = np.nonzero(w[k])
                if len(nonzero_indices[0]) > 0:
                    r, c = nonzero_indices[0][0], nonzero_indices[1][0]
                    val_in_matrix = remainder[r,c]
                    val_in_w = w[k][r,c]

                    for scalar in range(5):
                        if (val_in_w * scalar) % 5 == val_in_matrix:
                            coeffs[k-1] = scalar
                            remainder = mod5(remainder- mod5(scalar * w[k]))
                            break

            terms = [] 
            for idx, c in enumerate(coeffs):
                if c != 0:
                    if c==4:
                        terms.append(f"-omega_{idx+1}")
                    else:
                        terms.append(f"{c}*omega_{idx+1}")
            if terms:
                right_hand_side = " + ".join(terms).replace("+ -", "- ")
                print(f"omega_{i} * omega_{j} = {right_hand_side}")