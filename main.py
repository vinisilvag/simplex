import numpy as np


def zero(val):
    if abs(val) <= 1e-4:
        return 0

    return val


def read_input():
    n, m = map(int, input().split())

    A = []
    c = np.array(list(map(int, input().split())))

    for i in range(n):
        A.append(list(map(int, input().split())))

    A = np.array(A)

    b = A[:, m]
    A = A[:, :m]

    return n, m, A, b, c


def standard_equality_form(n, m, A, b, c):
    m += n

    A = np.concatenate((A, np.eye(n)), axis=1)

    for i in range(n):
        if b[i] < 0:
            b[i] *= -1
            A[i, :] *= -1

    c = np.concatenate((c, np.zeros(n)))

    b = np.reshape(b, (n, 1))
    c = np.reshape(c, (1, m))

    return n, m, A, b, c


def canonical(T, bases):
    n = T.shape[0] - 1

    for i in range(n):
        T[bases[i][0], :] /= T[bases[i][0]][bases[i][1]]

        for j in range(T.shape[0]):
            if j == bases[i][0]:
                continue

            T[j, :] -= T[bases[i][0], :] * T[j][bases[i][1]]

    return T


def simplex_iteration(T, bases):
    while np.any(T[0, (T.shape[0] - 1):(T.shape[1] - 1)] < 0):
        pass

    return T, bases


def simplex(n, m, A, b, c):
    # VERO
    aux_T = np.vstack((np.zeros(n), np.eye(n)))

    aux_T = np.hstack((
        aux_T,
        np.vstack((np.zeros(m), A)),  # A
        np.vstack((np.ones(n), np.eye(n))),  # auxiliary variables
        np.vstack((np.zeros(1), b))  # b
    ))

    print(aux_T)

    rows = list(range(1, n+1))
    columns = list(range(n + m, n + m + n))

    bases = np.array(list((rows[i], columns[i])
                     for i in range(n)), dtype="i,i")

    aux_T = canonical(aux_T, bases)

    print(aux_T)

    aux_T, bases = simplex_iteration(aux_T, bases)

    print(aux_T)
    print(bases)


def main():
    n, m, A, b, c = read_input()

    n, m, A, b, c = standard_equality_form(n, m, A, b, c)

    simplex(n, m, A, b, c)


if __name__ == "__main__":
    main()
