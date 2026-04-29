namespace Lab2.Core;

public static class Lab2Variant6
{
    public static SecondOrderProblem Create(double b, int n) =>
        b >= 0.75
            ? throw new ArgumentException("For this variant exact solution is defined for x < 0.75.")
            : new(
                SecondDerivative: (_, y, yPrime) =>
                    Math.Abs(y) < 1e-12
                        ? throw new InvalidOperationException("y is too close to zero.")
                        : (yPrime * yPrime) / y + (4.0 / 3.0) * Math.Pow(yPrime, 5),
                ExactSolution: x => Math.Pow(1.0 - 4.0 * x / 3.0, 0.75),
                X0: 0.0,
                Y0: 1.0,
                YPrime0: -1.0,
                B: b,
                N: n
            );
}
