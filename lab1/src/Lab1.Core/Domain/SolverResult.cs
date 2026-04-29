namespace Lab1.Core;

public sealed record SolverResult(
    string MethodName,
    IReadOnlyList<SolutionPoint> Points,
    double AbsoluteError,
    double RelativeError
);
