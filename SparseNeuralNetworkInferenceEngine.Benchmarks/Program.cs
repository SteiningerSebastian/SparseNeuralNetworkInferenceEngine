using BenchmarkDotNet.Running;
using System.Threading;

namespace SparseNeuralNetworkInferenceEngine.Benchmarks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;

            var _ = BenchmarkRunner.Run(typeof(Program).Assembly);
        }
    }
}
