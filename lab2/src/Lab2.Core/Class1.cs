using System.Globalization;
using System.Text;

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

public sealed record SolutionPoint(double X, double Approximation, double Exact);

public sealed record SolverResult(
    string MethodName,
    IReadOnlyList<SolutionPoint> Points,
    double AbsoluteError,
    double RelativeError
);

public interface ISecondOrderMethod
{
    string Name { get; }
    (double yNext, double yPrimeNext) Next(
        double x,
        double y,
        double yPrime,
        double h,
        Func<double, double, double, double> secondDerivative
    );
}

public sealed class RungeKutta4SecondOrderMethod : ISecondOrderMethod
{
    public string Name => "RungeKutta4";

    public (double yNext, double yPrimeNext) Next(
        double x,
        double y,
        double yPrime,
        double h,
        Func<double, double, double, double> secondDerivative
    )
    {
        double F1(double xx, double yy, double vv) => vv;
        double F2(double xx, double yy, double vv) => secondDerivative(xx, yy, vv);

        var k1y = F1(x, y, yPrime);
        var k1v = F2(x, y, yPrime);

        var k2y = F1(x + h / 2.0, y + h * k1y / 2.0, yPrime + h * k1v / 2.0);
        var k2v = F2(x + h / 2.0, y + h * k1y / 2.0, yPrime + h * k1v / 2.0);

        var k3y = F1(x + h / 2.0, y + h * k2y / 2.0, yPrime + h * k2v / 2.0);
        var k3v = F2(x + h / 2.0, y + h * k2y / 2.0, yPrime + h * k2v / 2.0);

        var k4y = F1(x + h, y + h * k3y, yPrime + h * k3v);
        var k4v = F2(x + h, y + h * k3y, yPrime + h * k3v);

        var yNext = y + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y);
        var yPrimeNext = yPrime + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v);
        return (yNext, yPrimeNext);
    }
}

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
        var v = problem.YPrime0;
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
                (y, v) = _method.Next(x, y, v, h, problem.SecondDerivative);
                x += h;
            }
        }

        return new SolverResult(_method.Name, points, maxAbs, maxRel);
    }
}

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

public interface IResultWriter
{
    void Write(string path, SolverResult result);
}

public sealed class CsvResultWriter : IResultWriter
{
    public void Write(string path, SolverResult result)
    {
        var sb = new StringBuilder();
        sb.AppendLine("method,x,approx,exact,abs_error,rel_error");
        foreach (var p in result.Points)
        {
            var abs = Math.Abs(p.Exact - p.Approximation);
            var rel = Math.Abs(p.Approximation) < 1e-12 ? 0.0 : abs / Math.Abs(p.Approximation);
            sb.AppendLine(
                string.Join(
                    ",",
                    result.MethodName,
                    p.X.ToString("G17", CultureInfo.InvariantCulture),
                    p.Approximation.ToString("G17", CultureInfo.InvariantCulture),
                    p.Exact.ToString("G17", CultureInfo.InvariantCulture),
                    abs.ToString("G17", CultureInfo.InvariantCulture),
                    rel.ToString("G17", CultureInfo.InvariantCulture)
                )
            );
        }

        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
    }
}
