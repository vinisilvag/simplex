import numpy as np


def zero(val):
    if abs(val) < 1e-4:
        val = 0.0

    return val


def print_array(arr):
    for i in arr:
        print("{:.7f}".format(i), end=' ')

    print()


def read_input():
    n, m = map(int, input().split())

    A = []
    c = np.array(list(map(int, input().split())))

    for i in range(n):
        A.append(list(map(int, input().split())))

    A = np.array(A)

    b = A[:, m]
    A = A[:, :m]

    A, b, c = A.astype(float), b.astype(float), c.astype(float)

    return n, m, A, b, c


def standard_equality_form(n, m, A, b, c):
    m += n

    A = np.concatenate((A, np.eye(n)), axis=1)

    invert = []

    for i in range(n):
        if b[i] < 0:
            invert.append(i + 1)
            b[i] = -b[i]
            A[i, :] = -A[i, :]

    c = np.concatenate((c, np.zeros(n)))

    b = np.reshape(b, (n, 1))
    c = np.reshape(c, (1, m))

    return n, m, invert, A, b, c


def canonical(T, i, j):
    if zero(T[i, j]) != 0:
        T[i, :] /= T[i, j]

    for k in range(T.shape[0]):
        if i == k:
            continue

        T[k, :] -= (T[i, :] * T[k, j])


def simplex_iteration(T, bases):
    while True:
        improving_col = -1

        for i in range(T.shape[0] - 1, T.shape[1] - 1):
            if T[0][i] < 0:
                improving_col = i
                break

        if improving_col == -1:
            break

        pivot_row = -1
        min_val = float('inf')

        for i in range(1, T.shape[0]):
            if T[i][improving_col] <= 0:
                continue

            if T[i][-1] / T[i][improving_col] < min_val:
                min_val = T[i][-1] / T[i][improving_col]
                pivot_row = i

        if pivot_row == -1:
            return True, improving_col, T, bases

        bases[pivot_row - 1] = (pivot_row, improving_col)

        canonical(T, pivot_row, improving_col)

    return False, -1, T, bases


def simplex(n, m, invert, A, b, c):
    aux_T = np.vstack((np.zeros(n), np.eye(n)))  # VERO

    aux_T = np.hstack((
        aux_T,
        np.vstack((np.zeros(m), A)),  # A
        np.vstack((np.ones(n), np.eye(n))),  # auxiliary variables
        np.vstack((np.zeros(1), b))  # b
    ))

    for i in invert:
        aux_T[i, :n] = -aux_T[i, :n]

    rows = list(range(1, n+1))
    columns = list(range(n + m, n + m + n))

    bases = np.array(list((rows[i], columns[i])
                     for i in range(n)), dtype="i,i")

    for i, j in bases:
        canonical(aux_T, i, j)

    _, _, aux_T, bases = simplex_iteration(aux_T, bases)
    optimal = aux_T[0, -1]

    print(aux_T)

    if optimal < 0:
        return "inviavel", (), (aux_T[0, :n]), ()
    else:
        T = np.vstack((np.zeros(n), np.eye(n)))  # VERO

        T = np.hstack((
            T,
            np.vstack((-c, A)),  # A
            np.vstack((np.zeros(1), b))  # b
        ))

        for i in invert:
            T[i, :n] = -T[i, :n]

        for i, j in bases:
            if j < n + m:
                canonical(T, i, j)

        unbounded, unbounded_col, T, bases = simplex_iteration(T, bases)

        if unbounded:
            x = np.zeros((m - n))

            for row, col in bases:
                if col - n >= 0 and col - n < (m - n):
                    x[col - n] = T[row, -1]

            certificate = np.zeros((m - n))

            if unbounded_col - n >= 0 and unbounded_col - n < (m - n):
                certificate[unbounded_col - n] = 1

            for row, col in bases:
                if col - n >= 0 and col - n < (m - n):
                    certificate[col - n] = -T[row, unbounded_col]

            return "ilimitada", (), (), (x, certificate)
        else:
            x = np.zeros((m - n))

            for row, col in bases:
                if col - n >= 0 and col - n < (m - n):
                    x[col - n] = T[row, -1]

            return "otima", (T[0, -1], x, T[0, :n]), (), ()


def main():
    n, m, A, b, c = read_input()

    n, m, invert, A, b, c = standard_equality_form(n, m, A, b, c)

    classification, optimum, infeasible, unbounded = simplex(
        n, m, invert, A, b, c
    )

    print(classification)

    if classification == "otima":
        optimal, x, certificate = optimum

        print("{:.7f}".format(optimal))
        print_array(x)
        print_array(certificate)
    elif classification == "inviavel":
        certificate = infeasible

        print_array(certificate)
    else:
        x, certificate = unbounded

        print_array(x)
        print_array(certificate)


if __name__ == "__main__":
    main()
