import math


def _sign(x):
    # Note that the sign function is undefined at x == 0
    # In order to make it differentiable, we will take a subgradient of 0 at x == 0
    return 1 if x > 0 else 0 if x == 0 else -1


def mae(
    xs: list[float], ys: list[float], actuals: list[float]
) -> tuple[float, float, float]:
    """
    Calculates Mean Absolute Error (MAE) for a vector of predictions vs. expectations.
    Also calculates the gradient for m and b based on the MAE

    Parameters:
        xs (list[float]):
            The vector of inputs
        ys (list[float]):
            The vector of expected outputs for each input
        actuals (list[float]):
            The vector of actual predictions for each input

    Returns:
        tuple[float, float, float]:
            The MAE, gradient for m, and gradient for b
    """
    error = 0
    delta_m = 0
    delta_b = 0

    for i in range(len(actuals)):
        error += abs(ys[i] - actuals[i])
        delta_m += xs[i] * _sign(ys[i] - actuals[i])
        delta_b = _sign(ys[i] - actuals[i])

    error = error * (1 / len(actuals))
    delta_m = -delta_m / len(actuals)
    delta_b = -delta_b / len(actuals)

    return error, delta_m, delta_b


def rmse(
    xs: list[float], ys: list[float], actuals: list[float]
) -> tuple[float, float, float]:
    """
    Calculates Root Mean Squared Error (RMSE) for a vector of predictions vs. expectations.
    Also calculates the gradient for m and b based on the RMSE

    Parameters:
        xs (list[float]):
            The vector of inputs
        ys (list[float]):
            The vector of expected outputs for each input
        actuals (list[float]):
            The vector of actual predictions for each input

    Returns:
        tuple[float, float, float]:
            The RMSE, gradient for m, and gradient for b
    """
    error = 0
    delta_m = 0
    delta_b = 0

    for i in range(len(actuals)):
        error += (ys[i] - actuals[i]) ** 2
        delta_m += xs[i] * (ys[i] - actuals[i])
        delta_b = ys[i] - actuals[i]

    error = math.sqrt(error * (1 / len(actuals)))
    delta_m = -delta_m / (len(actuals) * error)
    delta_b = -delta_b / (len(actuals) * error)

    return error, delta_m, delta_b


def mse(
    xs: list[float], ys: list[float], actuals: list[float]
) -> tuple[float, float, float]:
    """
    Calculates Mean Squared Error (MSE) for a vector of predictions vs. expectations.
    Also calculates the gradient for m and b based on the MSE

    Parameters:
        xs (list[float]):
            The vector of inputs
        ys (list[float]):
            The vector of expected outputs for each input
        actuals (list[float]):
            The vector of actual predictions for each input

    Returns:
        tuple[float, float, float]:
            The MSE, gradient for m, and gradient for b
    """
    error = 0
    delta_m = 0
    delta_b = 0

    for i in range(len(actuals)):
        error += (ys[i] - actuals[i]) ** 2
        delta_m += xs[i] * (ys[i] - actuals[i])
        delta_b += ys[i] - actuals[i]

    error = error * (1 / len(actuals))
    delta_m = delta_m * (-2 / len(actuals))
    delta_b = delta_b * (-2 / len(actuals))

    return error, delta_m, delta_b
