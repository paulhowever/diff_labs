using Lab2.Core;

namespace Lab2.Tests;

public class NumericalMethodTests
{
    [Theory]
    [InlineData(0.6, 200)]
    [InlineData(0.7, 300)]
    public void Rk4_ShouldConvergeForVariant6(double b, int n)
    {
        var problem = Lab2Variant6.Create(b, n);
        var result = new SecondOrderSolver(new RungeKutta4SecondOrderMethod()).Solve(problem);
        Assert.True(result.AbsoluteError < 5e-5);
    }

    [Fact]
    public void RelativeError_ShouldBeFinite()
    {
        var problem = Lab2Variant6.Create(0.7, 200);
        var result = new SecondOrderSolver(new RungeKutta4SecondOrderMethod()).Solve(problem);
        Assert.False(double.IsNaN(result.RelativeError));
        Assert.False(double.IsInfinity(result.RelativeError));
    }
}
