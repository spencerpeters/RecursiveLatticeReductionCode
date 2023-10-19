import os
import numpy as np
import math
from functools import cache
from collections import namedtuple
from numba import jit

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import ScalarFormatter

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "mathptmx"


### PRELIMINARIES ###

def blichfeldt_bound_on_lambda1(n):
    return math.sqrt((2 / np.pi) * math.gamma(2 + n / 2) ** (2 / n))


def svp_bound_on_gamma(n):
    if n == 1:
        return math.sqrt(1 ** (1 / n))
    elif n == 2:
        return math.sqrt((4 / 3) ** (1 / n))
    elif n == 3:
        return math.sqrt(2 ** (1 / n))
    elif n == 4:
        return math.sqrt(4 ** (1 / n))
    elif n == 5:
        return math.sqrt(8 ** (1 / n))
    elif n == 6:
        return math.sqrt((64 / 3) ** (1 / n))
    elif n == 7:
        return math.sqrt(64 ** (1 / n))
    elif n == 8:
        return math.sqrt(256 ** (1 / n))
    else:
        return blichfeldt_bound_on_lambda1(n)


def lll_bound_on_log_gamma(n, l):
    return (np.log2(4 / 3) / 4) * l * (n - l)

### WORKING WITH COARSE INDICES ###

# Of the functions in this section, only this function is used for generating the results.
@jit  # Just-in-time compilation for speed of main dynamic programming loop
def C_index_of(base, exponent, digit):
    return (base - 1) * exponent + digit


# The remaining 4 functions are only used for plotting.
def get_exponent(base, C_index):
    if C_index == 0:
        return 0
    exponent = (C_index - 1) // (base - 1)
    return exponent


def get_digit(base, C_index):
    if C_index == 0:
        return 0
    return (C_index - 1) % (base - 1) + 1


def get_value_of(base, C_index):
    exponent = get_exponent(base, C_index)
    digit = get_digit(base, C_index)
    return digit * base ** exponent


def get_log2_value_of(base, C_index):
    exponent = get_exponent(base, C_index)
    digit = get_digit(base, C_index)
    return exponent * np.log2(base) + np.log2(digit)

# If base = 2**power_of_two, the C_indices returned correspond to a geometric series of C values.
def logspaced_indices(power_of_two, array_length):
    indices = []
    i = 1
    while i < array_length:
        current_exponent = 0
        while current_exponent < power_of_two:
            i += 2 ** current_exponent
            if i >= array_length:
                return indices
            indices.append(i)
            current_exponent += 1
    return indices


### RECURRENCE SOLVER ###

# For 1 <= n <= max_N, 1 <= l <= n, 0 <= C_index <= (base - 1) * max_exponent,
# out[n, l, C_index] is the solution \gamma(n, l, C) to the coarse-grained
# recurrence reducing to SVP calls in dimension k, where C = get_value_of(C_index).
def solve_fixed_k_recurrence(max_N, base, max_exponent, k):
    max_C_index = (base - 1) * max_exponent
    results = np.full((max_N + 1, max_N + 1, max_C_index + 1), np.inf)

    # LLL base case for C_index = 0
    for n in range(k, max_N + 1):
        for l in range(1, n):
            results[n, l, :] = lll_bound_on_log_gamma(n, l)

    # SVP base case for n = k, l = 1 and n = k, l = k - 1
    results[k, 1, 1:] = np.log2(svp_bound_on_gamma(k))
    results[k, k - 1, 1:] = np.log2(svp_bound_on_gamma(k))

    return dp_loop(results, max_N, base, max_exponent)

# For 1 <= n <= max_N, 1 <= l <= n, 0 <= T_index <= (base - 1) * max_exponent,
# out[n, l, T_index] is the solution \gamma(n, l, T) to the coarse-grained
# recurrence reducing to SVP calls in arbitrary dimension, where T = get_value_of(T_index).
def solve_variable_k_recurrence(max_N, base, max_exponent):
    max_C_index = (base - 1) * max_exponent
    results = np.full((max_N + 1, max_N + 1, max_C_index + 1), np.inf)

    # LLL base case for C_index = 0
    for n in range(1, max_N + 1):
        for l in range(1, n):
            # results[n, l, :] = (np.log2(4 / 3) / 4) * l * (n - l)
            results[n, l, 0] = (np.log2(4 / 3) / 4) * l * (n - l)


    # SVP base cases
    for n in range(1, max_N):
        for C_index in range(1, max_C_index + 1):
            # logC = np.log2(get_value_of(base, C_index))
            logC = get_log2_value_of(base, C_index)
            if logC >= n:
                results[n, 1, C_index] = np.log2(svp_bound_on_gamma(n))

    return dp_loop(results, max_N, base, max_exponent)


# The main dynamic programming loop.
@jit  # Just-in-time compilation for speed of main dynamic programming loop
def dp_loop(results, max_N, base, max_exponent):
    for n in range(1, max_N + 1):
        for exponent in range(0, max_exponent):
            for digit in range(1, base):
                current_C_index = C_index_of(base, exponent, digit)
                for l_hat in range(1, n // 2 + 1):
                    best_so_far = min(results[n, l_hat, current_C_index], results[n, n - l_hat, current_C_index])
                    for l in [l_hat, n - l_hat]:
                        for l_star in range(1, n - l):
                            # Each recursive call either has smaller n or smaller runtime.
                            # We don't allow the reduction to allocate all oracle calls to the right branch
                            # (C_star = C) but we do allow C_star = 0.
                            # C_star = 0 case
                            best_so_far = min(best_so_far,
                                              results[n - l_star, l, current_C_index] + (
                                                      l / (n - l_star)) * results[
                                                  n, l_star, 0])

                            # digit = 1 case (we reduce to C values with exponent' = exponent - 1)
                            if digit == 1 and exponent > 0:
                                for left_digit in range(1, base):
                                    right_digit = base - left_digit
                                    left_index = C_index_of(base, exponent - 1, left_digit)
                                    right_index = C_index_of(base, exponent - 1, right_digit)
                                    best_so_far = min(best_so_far,
                                                      results[n - l_star, l, left_index] + (
                                                              l / (n - l_star)) * results[
                                                          n, l_star, right_index])

                            # digit > 1 case (we reduce to C values with exponent' = exponent, but digit' < digit)
                            elif digit > 1:
                                for left_digit in range(1, digit):
                                    right_digit = digit - left_digit
                                    left_index = C_index_of(base, exponent, left_digit)
                                    right_index = C_index_of(base, exponent, right_digit)
                                    best_so_far = min(best_so_far,
                                                      results[n - l_star, l, left_index] + (
                                                              l / (n - l_star)) * results[
                                                          n, l_star, right_index])
                    results[n, l_hat, current_C_index] = best_so_far
                    results[n, n - l_hat, current_C_index] = best_so_far
    return results


### THEOREM 5.1 ###

TradeoffPoint = namedtuple("TradeoffPoint", "alpha, C")


@cache
def theorem_51_tradeoff_point(k, n, l, tau):
    if max(1, (n - k) / 5) < l < n / 2 or l >= n - max(1, (n - k) / 10):
        return theorem_51_tradeoff_point(k, n, n - l, tau)

    if tau == 0:
        return TradeoffPoint(alpha=((np.log(4 / 3) / np.log(2)) / 4) * l * (n - l), C=0)
    elif n == k:
        assert l == 1
        return TradeoffPoint(alpha=np.log2(svp_bound_on_gamma(k)), C=1)
    elif l >= n / 2:
        lstar = math.ceil((n - k) / 20)
        t_left = theorem_51_tradeoff_point(k, n - lstar, l, tau)
        t_right = theorem_51_tradeoff_point(k, n, lstar, tau)
        return TradeoffPoint(alpha=t_left.alpha + (l / (n - lstar)) * t_right.alpha, C=t_left.C + t_right.C)
    else:
        lstar = math.ceil((n - k) / 20)
        t_left = theorem_51_tradeoff_point(k, n - lstar, l, tau)
        t_right = theorem_51_tradeoff_point(k, n, lstar, tau - 1)
        return TradeoffPoint(alpha=t_left.alpha + (l / (n - lstar)) * t_right.alpha, C=t_left.C + t_right.C)


### MICHAEL WALTER UPPER BOUND ###

def slide_oracle_calls_ub(epsilon, k, n, l=1):
    prefactor = n * n * n / (2 * k * l * (k - l))
    fixed_costs = prefactor * np.log(n * n / (2 * k) + n * n * n * np.log(blichfeldt_bound_on_lambda1(k)) / (4 * k * k * k))
    variable_costs = prefactor * np.log(1 / epsilon)
    return fixed_costs + variable_costs

### PLOTTING CODE ###

def fixed_k_plot_n_is_50():
    fixed_k_plot(50, 10, 2000, 20000, 10.3, 11.3, 3.5, [10.2, 10.4, 10.6, 10.8, 11, 11.1],
                 ["", "10.4", "", "", "", "11.2"])


def fixed_k_plot_n_is_100():
    fixed_k_plot(100, 30, 2000, 20000, 14.5, 17, 5, [14.5, 15, 15.5, 16, 16.5, 17], ["", "15", "", "", "", "17"])


def fixed_k_plot(n, k, x1, x2, y1, y2, zoom, ytickvalues, yticklabels):
    ### PLOT PARAMETERS ###
    base = 8
    max_exp = 8
    power_of_two = 3
    eps_ub = 0.2
    max_tau = 8
    max_C_index = (base - 1) * max_exp

    ### CURVES TO PLOT ###

    # Recursive algorithm with optimal parameters
    optimal_tradeoffs = solve_fixed_k_recurrence(n, base, max_exp, k)
    svp_calls_by_C_index = np.asarray([get_value_of(base, i) for i in range(max_C_index + 1)])
    recursive_gammas = 2 ** optimal_tradeoffs[n, 1, :]

    # Concrete recursive algorithm of Theorem 5.1
    theorem_51_points = [theorem_51_tradeoff_point(k, n, 1, i) for i in range(1, max_tau)]
    theorem_51_gammas = [2 ** t.alpha for t in theorem_51_points]
    theorem_51_svp_calls = [t.C for t in theorem_51_points]

    # Upper bound on slide reduction due to Michael Walter
    epsilons = np.logspace(eps_ub, -7, 100)
    mw_ub_svp_calls = [slide_oracle_calls_ub(epsilon, k, n) for epsilon in epsilons]
    mw_ub_gammas = [(1 + epsilon) * blichfeldt_bound_on_lambda1(k) ** ((n - 1) / (k - 1)) for epsilon in epsilons]

    ### MAIN PLOT ###

    main_axes = plt.axes()

    def plot_all(axes):
        # Plot concrete recursive algorithm
        axes.loglog(theorem_51_svp_calls,
                    theorem_51_gammas,
                    nonpositive="mask",
                    label="Recursive, Section 5",
                    linestyle="None",
                    marker="*",
                    ms=6,
                    color="red")

        # Plot recursive algorithm with optimal parameters
        axes.loglog(svp_calls_by_C_index[logspaced_indices(power_of_two, len(svp_calls_by_C_index))],
                    recursive_gammas[logspaced_indices(power_of_two, len(recursive_gammas))],
                    label="Recursive, optimal parameters",
                    nonpositive="mask",
                    linestyle="dashed",
                    marker=".")

        # Plot slide reduction upper bound
        axes.loglog(mw_ub_svp_calls[10:], mw_ub_gammas[10:], label="Upper bound on slide reduction")

        # Plot limiting approximation factor from Theorem 5.1
        axes.loglog(svp_calls_by_C_index,
                    [svp_bound_on_gamma(k) ** ((n - 1) / (k - 1))] * len(svp_calls_by_C_index),
                    label=r"Value as $C \to \infty$",
                    linestyle="dashed",
                    color="green")

    plot_all(main_axes)
    # Finalize the main plot
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
    plt.legend()
    plt.title(fr"Comparison of $\gamma$-SVP tradeoffs for $n = {n}, k = {k}$")
    plt.xlabel(r"Number of oracle calls $C$")
    plt.ylabel(r"HSVP approximation factor $\gamma$")

    ### INSET PLOT ###

    inset_axes = zoomed_inset_axes(main_axes, zoom, 'center right')
    plot_all(inset_axes)

    # Finalize inset plot
    inset_axes.set_xlim(x1, x2)
    inset_axes.set_ylim(y1, y2)
    inset_axes.xaxis.set_ticklabels([], minor=True)

    inset_axes.spines["left"].set_color("gray")
    inset_axes.spines["right"].set_color("gray")
    inset_axes.spines["top"].set_color("gray")
    inset_axes.spines["bottom"].set_color("gray")
    mark_inset(main_axes, inset_axes, loc1=2, loc2=4, fc="none", ec="gray", linestyle="")

    inset_axes.set_yticks(ytickvalues, yticklabels, minor=True)

    plt.draw()
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    plt.savefig(f"plots/fixed_k_n_is_{n}.png", bbox_inches="tight")
    plt.show()


def variable_k_plot_n_is_50():
    variable_k_plot(50, 18, [5, 10, 20, 30, 40, 49])


def variable_k_plot_n_is_100():
    variable_k_plot(100, 35, [5, 10, 20, 40, 60, 80, 90, 99])


def variable_k_plot(n, max_exponent, ks):
    base = 8
    power_of_two = 3
    max_C_index = (base - 1) * max_exponent

    ### CURVES TO PLOT ###
    variable_reduction = solve_variable_k_recurrence(n, base, max_exponent)
    fixed_reductions = []
    for k in ks:
        fixed_reductions.append(solve_fixed_k_recurrence(n, base, max_exponent, k))

    ### PLOTTING CODE ###
    recursive_gammas = 2 ** variable_reduction[n, 1, :]
    running_time = np.asarray([get_value_of(base, i) for i in range(max_C_index + 1)])
    plt.loglog(running_time[logspaced_indices(power_of_two, len(running_time))],
               recursive_gammas[logspaced_indices(power_of_two, len(recursive_gammas))],
               label="Any $k$", linestyle="dashed", marker=".")

    for j in range(len(ks)):
        k = ks[j]
        recurrence_to_k = fixed_reductions[j]

        running_time = np.asarray([2 ** k * get_value_of(base, i) for i in range(max_C_index + 1)])
        # Enforce that all curves stop at running time corresponding to max_C_index
        running_time = np.asarray([r for r in running_time if r <= get_value_of(base, max_C_index)])

        recursive_gammas = 2 ** recurrence_to_k[n, 1, :]
        # Don't display trivial approximation factors from reductions without enough running time
        # to make a single oracle call.
        recursive_gammas = np.asarray([np.inf if running_time[i] < 2 ** k else recursive_gammas[i] for i in
                                       range(len(running_time))])
        plt.loglog(running_time[logspaced_indices(power_of_two, len(running_time))],
                   recursive_gammas[logspaced_indices(power_of_two, len(recursive_gammas))],
                   nonpositive="mask", label=rf"Fixed $k = {k}$",
                   linestyle="dashed", marker=".")

    # Finalize plot
    plt.xlabel(r"Running time $T$")
    plt.ylabel(r"HSVP approximation factor $\gamma$")
    plt.title(fr"Reducing to fixed vs. variable dimension, $n = {n}$.")
    plt.legend(loc="lower left")
    plt.draw()
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    plt.savefig(f"plots/variable_k_n_is_{n}.png", bbox_inches="tight")
    plt.show()


### GENERATE PLOTS ###

if __name__ == "__main__":
    fixed_k_plot_n_is_50()
    fixed_k_plot_n_is_100()
    variable_k_plot_n_is_50()
    variable_k_plot_n_is_100()
