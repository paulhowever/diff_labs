namespace Lab1.Core;

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
