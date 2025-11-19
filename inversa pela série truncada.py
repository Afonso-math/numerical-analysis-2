import numpy as np

def hilbert_matrix(n):
    matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1/(i + j + 1)
    return matrix

def norm_one(A):
    cols_sum = np.sum(np.abs(A), axis=0)
    return np.max(cols_sum)

def norm_infinite(A):
    max_row_sum = 0
    rows, cols = A.shape
    for i in range(rows):
        row_sum = 0
        for j in range(cols):
            row_sum += abs(A[i][j])
        if row_sum > max_row_sum:
            max_row_sum = row_sum
    return max_row_sum

def B_matrix(A):
    alfa = 10**(-16) + norm_infinite(A) * norm_infinite(A.T)
    B = A.T / alfa
    return B

def compute_inverse_series(A, B_init, k):
    rows, cols = A.shape
    I = np.eye(rows)
    inverse_A = np.zeros_like(A)

    for j in range(k + 1):
        inverse_A += B_init @ np.linalg.matrix_power(I - A @ B_init, j)
    return inverse_A

def erro(A_inv, A):
    E = A_inv @ A - np.eye(A.shape[0])
    return np.max(np.sum(np.abs(E), axis=1))
def main():
    n = int(input("Insira a dimensão da matriz de Hilbert: "))
    iteradas = int(input("Insira o número de iteradas desejado: "))
    A = hilbert_matrix(n)
    B_inicial = B_matrix(A)
    inverted_A = compute_inverse_series(A, B_inicial, iteradas)
    A_inv_LU = np.linalg.inv(A)
    print(f"Matriz de Hilbert com dimensão {n}: {A}")
    print(f"Inversa da matriz de Hilbert com série de Neumann: {inverted_A}")
    print(f"Inversa da matriz de Hilbert com decomposição LU: {A_inv_LU}")
    erro_serie = erro(inverted_A, A)
    erro_LU = erro(A_inv_LU, A)
    print(f"Erro funcional - série de Neumann: {erro_serie}")
    print(f"Erro funcional - decomposição LU: {erro_LU}")

main()
