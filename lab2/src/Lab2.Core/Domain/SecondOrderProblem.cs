namespace Lab2.Core;

public sealed record SecondOrderProblem(
    Func<double, double, double, double> SecondDerivative,
    Func<double, double> ExactSolution,
    double X0,
    double Y0,
    double YPrime0,
    double B,
    int N
);
