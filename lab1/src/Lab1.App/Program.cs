using Lab1.Core;

var b = 2.0;
var n = 200;
var problem = Lab1Variant6.Create(b, n);

var methods = new IFirstOrderMethod[]
{
    new EulerMethod(),
    new HeunMethod(),
    new RungeKutta4Method()
};

var outputDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "data"));
Directory.CreateDirectory(outputDir);
var writer = new CsvResultWriter();

foreach (var method in methods)
{
    var solver = new FirstOrderSolver(method);
    var result = solver.Solve(problem);
    var path = Path.Combine(outputDir, $"{method.Name.ToLowerInvariant()}.csv");
    writer.Write(path, result);
    Console.WriteLine($"{method.Name}: abs={result.AbsoluteError:E3}, rel={result.RelativeError:E3}, file={path}");
}
