import ml_from_scratch.regression.loss as loss

LOSS_FUNCTIONS = ["MSE", "RMSE", "MAE"]


class linear_regression:
    """
    An implementation of a 2D linear regression model. The model stores
    the loss function being used, and the two parameters: m and b.
    """

    # Take xs and ys out of variables - just pass to training funcs
    loss: str
    m: float
    b: float

    def __init__(self, loss: str = "MSE"):
        if loss not in LOSS_FUNCTIONS:
            raise ValueError("Loss must be one of the following:", LOSS_FUNCTIONS)
        self.loss = loss
        self.m = 1
        self.b = 0

    def _apply_equation(self, x):
        return (self.m * x) + self.b

    def train_one_step(self, alpha: float, xs: list[float], ys: list[float]) -> float:
        """
        Proceed with one step of model training, yielding the error as calculated
        before the weight update for that step.

        Parameters:
            alpha (float):
                The learning rate (how much to update the parameters based on the gradient)
            xs (list[float]):
                The vector of inputs
            ys (list[float]):
                The vector of expected outputs for each input

        Returns:
            float:
                The error as calculated before the weight update occurred
        """
        actuals = [self._apply_equation(x) for x in xs]

        if self.loss == "MSE":
            error, delta_m, delta_b = loss.mse(xs, ys, actuals)
        elif self.loss == "RMSE":
            error, delta_m, delta_b = loss.rmse(xs, ys, actuals)
        elif self.loss == "MAE":
            error, delta_m, delta_b = loss.mae(xs, ys, actuals)

        self.m = self.m - (alpha * delta_m)
        self.b = self.b - (alpha * delta_b)

        return error

    def train(
        self,
        alpha: float,
        xs: list[float],
        ys: list[float],
        max_iterations: int = 100,
        error_change_cutoff: float = 0.001,
    ) -> None:
        """
        Proceeds with training until either max_iterations is reached, or
        the error is changing less than error_change_cutoff between iterations.

        Parameters:
            alpha (float):
                The learning rate (how much to update the parameters based on the gradient)
            xs (list[float]):
                The vector of inputs
            ys (list[float]):
                The vector of expected outputs for each input
            max_iterations (int, default = 100):
                The maximum number of training iterations to perform
            error_change_cutoff (float, default = 0.001):
                Training will stop if the change in error between iterations is less than this number
        """
        if len(xs) != len(ys):
            raise ValueError("Input arrays xs and ys must be the same length")

        prev_error = -1
        error = -1
        num_iterations = 0

        # Note - this training loop isn't perfect. The error reported for each step is the error
        # before the parameters are updated. So the error is always "one step behind"
        # For this example, that's fine
        while prev_error == -1 or num_iterations < max_iterations:
            if num_iterations >= 2 and abs(error - prev_error) < error_change_cutoff:
                break
            num_iterations += 1
            prev_error = error
            error = self.train_one_step(alpha, xs, ys)

        print("Training completed:")
        print("\tIterations:", num_iterations)
        print("\tFinal Error:", error)
        print(f"\tFinal Equation: y = {self.m}x + {self.b}")
