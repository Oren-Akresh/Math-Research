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
x[6,5] = 11

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


# COMPUTATION
# single check

for i in range(1, 9):
    for j in range(1, 8):
        x_wi = bracket(x, w[i])
        x_wj = bracket(x, w[j])

        matrix = mod5(bracket(x_wi, w[j]) - bracket(w[i], x_wj))

        if np.any(matrix != 0):
            coeffs = [0] * 8
            remainder = np.copy(matrix)

            for k in range(1, 9):
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
                    display_coeff = f"-{1 if c==4 else ''}" if c ==4 else f"{c}*"
                    if c==4:
                        terms.append(f"-omega_{idx+1}")
                    else:
                        terms.append(f"{c}*omega_{idx+1}")
            if terms:
                right_hand_side = " + ".join(terms).replace("+ -", "- ")
                print(f"omega_{i} * omega_{j} = {right_hand_side}")