namespace Lab2.Core;

public sealed class SecondOrderSolver
{
    private readonly ISecondOrderMethod _method;

    public SecondOrderSolver(ISecondOrderMethod method)
    {
        _method = method;
    }

    public SolverResult Solve(SecondOrderProblem problem)
    {
        if (problem.N <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(problem.N), "N must be greater than zero.");
        }

        if (problem.B <= problem.X0)
        {
            throw new ArgumentException("B must be greater than X0.");
        }

        var h = (problem.B - problem.X0) / problem.N;
        var points = new List<SolutionPoint>(problem.N + 1);
        var x = problem.X0;
        var y = problem.Y0;
        var yPrime = problem.YPrime0;
        var maxAbs = 0.0;
        var maxRel = 0.0;

        for (var k = 0; k <= problem.N; k++)
        {
            var exact = problem.ExactSolution(x);
            var absError = Math.Abs(exact - y);
            var relError = Math.Abs(y) < 1e-12 ? 0.0 : absError / Math.Abs(y);
            if (k > 0)
            {
                maxAbs = Math.Max(maxAbs, absError);
                maxRel = Math.Max(maxRel, relError);
            }
            points.Add(new SolutionPoint(x, y, exact));

            if (k < problem.N)
            {
                (y, yPrime) = _method.Next(x, y, yPrime, h, problem.SecondDerivative);
                x += h;
            }
        }

        return new SolverResult(_method.Name, points, maxAbs, maxRel);
    }
}
