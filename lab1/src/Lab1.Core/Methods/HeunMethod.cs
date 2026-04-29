namespace Lab1.Core;

public sealed class HeunMethod : IFirstOrderMethod
{
    public string Name => "Heun";

    public double Next(double x, double y, double h, Func<double, double, double> derivative)
    {
        var predictor = y + h * derivative(x, y);
        return y + 0.5 * h * (derivative(x, y) + derivative(x + h, predictor));
    }
}
