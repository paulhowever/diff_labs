using Lab1.Core;

namespace Lab1.Tests;

public class NumericalMethodTests
{
    [Theory]
    [InlineData(2.0, 200)]
    [InlineData(2.0, 400)]
    public void Rk4_ShouldHaveHighAccuracy(double b, int n)
    {
        var problem = Lab1Variant6.Create(b, n);
        var result = new FirstOrderSolver(new RungeKutta4Method()).Solve(problem);
        Assert.True(result.AbsoluteError < 1e-7);
    }

    [Fact]
    public void Heun_ShouldBeBetterThanEuler()
    {
        var problem = Lab1Variant6.Create(2.0, 200);
        var euler = new FirstOrderSolver(new EulerMethod()).Solve(problem);
        var heun = new FirstOrderSolver(new HeunMethod()).Solve(problem);
        Assert.True(heun.AbsoluteError < euler.AbsoluteError);
    }

    [Fact]
    public void RelativeError_ShouldBeFinite()
    {
        var problem = Lab1Variant6.Create(2.0, 200);
        var rk4 = new FirstOrderSolver(new RungeKutta4Method()).Solve(problem);
        Assert.False(double.IsNaN(rk4.RelativeError));
        Assert.False(double.IsInfinity(rk4.RelativeError));
    }
}
