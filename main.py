import numpy as np
import pandas as pd
from read_file import load_json_file
from markov_chain import create_transition_matrix, m_i, p_i, pi_i, population_n, flow_d, usage_u, waiting_time_w, loss, create_probability_vector, _pi_i


def indexes(_lambda, c, k, pi_is, m_is):
    population = population_n(k, pi_is)
    print('N = %.4f' % population)

    flow = flow_d(k, pi_is, m_is)
    print('D = %.4f' % flow)

    usage = usage_u(k, c, pi_is)
    print('U = %.4f' % usage)

    waiting_time = waiting_time_w(population, flow)
    print('W = %.4f' % waiting_time)

    if _lambda > 0:
        l = loss(_lambda, pi_is)
        print('Loss = %.4f' % l)


def solve_queue(_lambda, m, c, k, title):
    print('\n%s\n' % title)

    m_is = [0] * (k + 1)

    for i in range(k + 1):
        m_is[i] = m_i(i, c, m)
        print('M_%d = %.4f' % (i, m_is[i]))

    p_is = [0] * (k + 1)

    for i in range(k + 1):
        p_is[i] = p_i(i, _lambda, c, m, previous_p_i=None if i == 0 else p_is[i - 1], current_m_i=m_is[i])
        print('P_%d = %.4f' % (i, p_is[i]))

    sum_of_p_is = sum(p_is)
    print('sum P = %.4f' % sum_of_p_is)

    pi_is = [0] * (k + 1)

    for i in range(k + 1):
        pi_is[i] = pi_i(i, _lambda, c, m, sum_of_p_is, current_p_i=p_is[i])
        print('PI_%d = %.4f' % (i, pi_is[i]))

    indexes(_lambda, c, k, pi_is, m_is)


def example_1():
    solve_queue(2, 2, 3, 3, 'Example 1')


def example_2():
    print('\nExample 2\n')
    a = np.array([[-2.02, 2, 0, 0, 2], [2, -4.02, 4, 0, 0], [0, 2, -6.02, 6, 0], [0, 0, 2, -6.02, 0], [0.02, 0.02, 0.02, 0.02, -2]])
    df = pd.DataFrame(a)
    df = df.swapaxes(1, 0, copy=False)
    print(df)
    probability_vector = create_probability_vector(df)

    for i in range(probability_vector.size):
        print('P_%d = %.4f' % (i, probability_vector[i]))

    # Expected result [2.0469, 2.0134, 1, 0.3322, 0.0539]

    sum_of_probability_vector = sum(probability_vector)
    print('sum Probability Vector = %.4f' % sum_of_probability_vector)

    pi_is = [0] * probability_vector.size

    for i in range(probability_vector.size):
        pi_is[i] = _pi_i(probability_vector[i], sum_of_probability_vector)
        print('PI_%d = %.4f' % (i, pi_is[i]))

    sum_of_pi_is = sum(pi_is)
    print('sum PIs = %.4f' % sum_of_pi_is)


def example_3():
    print('\nExample 3\n')
    _lambda = 6
    m_a = 10
    m_d = 4
    c = 1
    k = 3

    markov_chain = load_json_file('markov_chain_example.json')
    transition_matrix = create_transition_matrix(markov_chain)

    print(transition_matrix)
    probability_vector = create_probability_vector(transition_matrix)

    for i in range(probability_vector.size):
        print('P_%d = %.4f' % (i, probability_vector[i]))

    sum_of_probability_vector = sum(probability_vector)
    print('sum Probability Vector = %.4f' % sum_of_probability_vector)

    pi_is = [0] * probability_vector.size

    for i in range(probability_vector.size):
        pi_is[i] = _pi_i(probability_vector[i], sum_of_probability_vector)
        print('PI_%d = %.4f' % (i, pi_is[i]))

    sum_of_pi_is = sum(pi_is)
    print('sum PIs = %.4f' % sum_of_pi_is)

    combined_m_is = [0] * (k + 1)

    for i in range(k + 1):
        m_a_i = m_i(i, c, m_a)
        m_d_i = m_i(i, c, m_d)
        combined_m_is[i] = m_a_i * m_d_i
        print('M_%d = %.4f' % (i, combined_m_is[i]))

    combined_pi_is = [0] * (k+1)

    for i in range(k+1):
        p_a_i = probability_vector[i]
        p_d_i = probability_vector[i+k+1]
        combined_pi_is[i] = p_a_i * p_d_i

    indexes(_lambda, c, k, combined_pi_is, combined_m_is)


def packaging_line():
    probabilities = {
        'fila_entrada': {
            'c': 1,
            'k': 1,
            'm': 1 / ((0.1 + 0.9) / 2),  # avg: 2 by minute
            'pi_is': [0.9375, 0.0625]
        },
        'fila_impressao': {
            'c': 1,
            'k': 10,
            'm': 1 / ((8 + 10) / 2),  # avg: 0.1111 by minute
            'pi_is': [0.1013, 0.4559, 0.3148, 0.0911, 0.0264, 0.0078, 0.0020, 0.0005, 0.0002, 0.0001, 0.00]
        },
        'fila_corte': {
            'c': 1,
            'k': 5,
            'm': 1 / ((2 + 5) / 2),  # avg: 0.2857 by minute
            'pi_is': [0.5492, 0.4024, 0.0471, 0.0013, 0.00, 0.00]
        },
        'fila_montagem': {
            'c': 1,
            'k': 3,
            'm': 1 / ((2 + 4) / 2),  # avg: 0.3333 by minute
            'pi_is': [0.6716, 0.3232, 0.0051, 0.00]
        }
    }

    for name, queue in probabilities.items():
        print(name)
        m_is = [0] * (queue['k'] + 1)

        for i in range(queue['k'] + 1):
            m_is[i] = m_i(i, queue['c'], queue['m'])

        indexes(0, queue['c'], queue['k'], queue['pi_is'], m_is)  # ignore the first param


if __name__ == '__main__':
    example_1()
    example_2()
    example_3()
    packaging_line()
