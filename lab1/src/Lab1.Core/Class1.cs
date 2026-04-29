using System.Globalization;
using System.Text;

namespace Lab1.Core;

public sealed record InitialValueProblem(
    Func<double, double, double> Derivative,
    Func<double, double> ExactSolution,
    double X0,
    double Y0,
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

public interface IFirstOrderMethod
{
    string Name { get; }
    double Next(double x, double y, double h, Func<double, double, double> derivative);
}

public sealed class EulerMethod : IFirstOrderMethod
{
    public string Name => "Euler";

    public double Next(double x, double y, double h, Func<double, double, double> derivative) =>
        y + h * derivative(x, y);
}

public sealed class HeunMethod : IFirstOrderMethod
{
    public string Name => "Heun";

    public double Next(double x, double y, double h, Func<double, double, double> derivative)
    {
        var predictor = y + h * derivative(x, y);
        return y + 0.5 * h * (derivative(x, y) + derivative(x + h, predictor));
    }
}

public sealed class RungeKutta4Method : IFirstOrderMethod
{
    public string Name => "RungeKutta4";

    public double Next(double x, double y, double h, Func<double, double, double> derivative)
    {
        var k1 = derivative(x, y);
        var k2 = derivative(x + h / 2.0, y + h * k1 / 2.0);
        var k3 = derivative(x + h / 2.0, y + h * k2 / 2.0);
        var k4 = derivative(x + h, y + h * k3);
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
}

public sealed class FirstOrderSolver
{
    private readonly IFirstOrderMethod _method;

    public FirstOrderSolver(IFirstOrderMethod method)
    {
        _method = method;
    }

    public SolverResult Solve(InitialValueProblem problem)
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
                y = _method.Next(x, y, h, problem.Derivative);
                x += h;
            }
        }

        return new SolverResult(_method.Name, points, maxAbs, maxRel);
    }
}

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
