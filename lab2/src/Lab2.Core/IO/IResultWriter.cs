namespace Lab2.Core;

public interface IResultWriter
{
    void Write(string path, SolverResult result);
}
