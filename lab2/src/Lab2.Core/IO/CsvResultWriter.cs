using System.Globalization;
using System.Text;

namespace Lab2.Core;

public sealed class CsvResultWriter : IResultWriter
{
    public void Write(string path, SolverResult result)
    {
        var sb = new StringBuilder();
        sb.AppendLine("method,x,approx,exact,abs_error,rel_error");
        foreach (var p in result.Points)
        {
            var abs = Math.Abs(p.Exact - p.Approximation);
            var rel = Math.Abs(p.Approximation) < 1e-12 ? 0.0 : abs / Math.Abs(p.Approximation);
            sb.AppendLine(
                string.Join(
                    ",",
                    result.MethodName,
                    p.X.ToString("G17", CultureInfo.InvariantCulture),
                    p.Approximation.ToString("G17", CultureInfo.InvariantCulture),
                    p.Exact.ToString("G17", CultureInfo.InvariantCulture),
                    abs.ToString("G17", CultureInfo.InvariantCulture),
                    rel.ToString("G17", CultureInfo.InvariantCulture)
                )
            );
        }

        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
    }
}
