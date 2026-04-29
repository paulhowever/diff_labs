namespace Lab2.Core;

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
