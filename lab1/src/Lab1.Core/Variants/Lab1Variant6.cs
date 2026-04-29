namespace Lab1.Core;

public static class Lab1Variant6
{
    public static InitialValueProblem Create(double b, int n) =>
        new(
            Derivative: (x, y) => 1.0 / (x * x + x + 1.0) - y / (x + 0.5),
            ExactSolution: x => (0.5 * Math.Log(x * x + x + 1.0) + 0.5) / (x + 0.5),
            X0: 0.0,
            Y0: 1.0,
            B: b,
            N: n
        );
}
