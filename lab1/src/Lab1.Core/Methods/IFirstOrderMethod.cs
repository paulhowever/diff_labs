namespace Lab1.Core;

public interface IFirstOrderMethod
{
    string Name { get; }
    double Next(double x, double y, double h, Func<double, double, double> derivative);
}
