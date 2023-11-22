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
    while True:
        improving_col = -1

        for i in range(T.shape[0] - 1, T.shape[1] - 1):
            if T[0][i] < 0:
                improving_col = i
                break

        if improving_col == -1:
            break

        pivot_row = -1
        min_val = np.finfo(np.float64).max

        for i in range(1, T.shape[0]):
            if T[i][improving_col] > 0 and T[i][-1:] / T[i][improving_col] < min_val:
                min_val = T[i][-1:] / T[i][improving_col]
                pivot_row = i

        if pivot_row == -1:
            return True, T, bases

        bases[pivot_row - 1] = (pivot_row, improving_col)

        canonical(T, bases)

    return False, T, bases


def simplex(n, m, A, b, c):
    # VERO
    aux_T = np.vstack((np.zeros(n), np.eye(n)))

    aux_T = np.hstack((
        aux_T,
        np.vstack((np.zeros(m), A)),  # A
        np.vstack((np.ones(n), np.eye(n))),  # auxiliary variables
        np.vstack((np.zeros(1), b))  # b
    ))

    rows = list(range(1, n+1))
    columns = list(range(n + m, n + m + n))

    bases = np.array(list((rows[i], columns[i])
                     for i in range(n)), dtype="i,i")

    aux_T = canonical(aux_T, bases)

    _, aux_T, bases = simplex_iteration(aux_T, bases)

    if aux_T[0, -1:] < 0:
        return "inviavel", (), (-aux_T[0, 0:n]), ()
    else:
        T = aux_T.copy()
        T = np.delete(T, list(range(n + m, n + m + n)), 1)
        T[0, n:n+m] = -c

        unbounded, T, bases = simplex_iteration(T, bases)

        if unbounded:
            return "ilimitada", (), (), ()
        else:
            x = np.zeros((m - n))

            for i in range(n):
                index = bases[i][1] - n
                if index >= 0 and index < m:
                    x[index] = T[bases[i][0], -1:]

            return "otima", (T[0, -1], x, T[0, 0:n]), (), ()


def main():
    n, m, A, b, c = read_input()

    n, m, A, b, c = standard_equality_form(n, m, A, b, c)

    classification, optimum, infeasible, unbounded = simplex(n, m, A, b, c)

    print(classification)

    if classification == "otima":
        optimal, x, certificate = optimum

        print(optimal)

        for i in x:
            print(i, end=' ')

        print()

        for i in certificate:
            print(i, end=' ')
    elif classification == "inviavel":
        certificate = infeasible

        for i in certificate:
            print(i, end=' ')
    else:
        pass

    print()


if __name__ == "__main__":
    main()
