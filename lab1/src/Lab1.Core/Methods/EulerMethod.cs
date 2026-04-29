namespace Lab1.Core;

public sealed class EulerMethod : IFirstOrderMethod
{
    public string Name => "Euler";

    public double Next(double x, double y, double h, Func<double, double, double> derivative) =>
        y + h * derivative(x, y);
}
