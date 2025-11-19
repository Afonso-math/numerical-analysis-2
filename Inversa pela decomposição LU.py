import numpy as np
from scipy.linalg import lu

def hilbert_matrix(n):
    matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1/(i + j + 1)
    return matrix

def LU_decomposition(A):
    P, L, U = lu(A)
    return P, L, U

def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])
    return y

def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i+1:] * x[i+1:])) / U[i, i]
    return x

def inverse_via_LU(A):
    P, L, U = lu(A)
    n = A.shape[0]
    A_inv = np.zeros_like(A, dtype=float)

    I = np.eye(n)
    for col in range(n):
        b = P @ I[:, col]
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)
        A_inv[:, col] = x
    return A_inv

def main():
    n = int(input("Insira a dimensão da matriz de Hilbert: "))
    A = hilbert_matrix(n)
    inversa = inverse_via_LU(A)
    print(f"Matriz de Hilbert com dimensão {n}: {A}")
    print(f"Inversa da matriz de Hilbert: {inversa}")

main()