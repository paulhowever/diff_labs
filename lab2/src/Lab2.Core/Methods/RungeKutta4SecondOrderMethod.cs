namespace Lab2.Core;

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
        double f1(double _, double __, double vv) => vv;
        double f2(double xx, double yy, double vv) => secondDerivative(xx, yy, vv);

        var k1y = f1(x, y, yPrime);
        var k1v = f2(x, y, yPrime);

        var k2y = f1(x + h / 2.0, y + h * k1y / 2.0, yPrime + h * k1v / 2.0);
        var k2v = f2(x + h / 2.0, y + h * k1y / 2.0, yPrime + h * k1v / 2.0);

        var k3y = f1(x + h / 2.0, y + h * k2y / 2.0, yPrime + h * k2v / 2.0);
        var k3v = f2(x + h / 2.0, y + h * k2y / 2.0, yPrime + h * k2v / 2.0);

        var k4y = f1(x + h, y + h * k3y, yPrime + h * k3v);
        var k4v = f2(x + h, y + h * k3y, yPrime + h * k3v);

        var yNext = y + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y);
        var yPrimeNext = yPrime + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v);
        return (yNext, yPrimeNext);
    }
}
