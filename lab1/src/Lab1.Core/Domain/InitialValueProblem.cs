namespace Lab1.Core;

public sealed record InitialValueProblem(
    Func<double, double, double> Derivative,
    Func<double, double> ExactSolution,
    double X0,
    double Y0,
    double B,
    int N
);
