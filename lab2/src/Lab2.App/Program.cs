using Lab2.Core;

var b = 0.7;
var n = 200;
var problem = Lab2Variant6.Create(b, n);
var solver = new SecondOrderSolver(new RungeKutta4SecondOrderMethod());
var result = solver.Solve(problem);

var outputDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "data"));
Directory.CreateDirectory(outputDir);
var path = Path.Combine(outputDir, "rungekutta4.csv");
new CsvResultWriter().Write(path, result);

Console.WriteLine($"RK4: abs={result.AbsoluteError:E3}, rel={result.RelativeError:E3}, file={path}");
