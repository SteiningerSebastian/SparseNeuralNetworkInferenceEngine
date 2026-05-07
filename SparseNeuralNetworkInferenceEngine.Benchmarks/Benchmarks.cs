using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using Microsoft.VSDiagnostics;

using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using SparseNeuralNetworkInferenceEngine.General;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.Collections.Generic;

namespace SparseNeuralNetworkInferenceEngine.Benchmarks
{
    [CPUUsageDiagnoser]
    [MemoryDiagnoser]
    [HardwareCounters(HardwareCounter.BranchMispredictions, HardwareCounter.CacheMisses)]
    public class Benchmarks
    {
        const int BATCH_SIZE = 128;

        protected InferenceSession session;
        protected List<NamedOnnxValue> inputs;

        [GlobalSetup(Target = nameof(OnnxMNISTDataset))]
        public void SetupOnnxMNIST()
        {
            //Load data / load the first 128 samples of the MNIST test set, which is stored as a binary file of floats (flattened images)
            float[] data = BinaryLoader.ReadFileToFloatEnumerable("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/Evaluation/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 784).ToArray();

            using var session = new InferenceSession("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/Evaluation/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/model.onnx");

            int[] dimensions = { BATCH_SIZE, 784 };
            var inputTensor = new DenseTensor<float>(data, dimensions);

            inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };
        }

        [Benchmark]
        public void OnnxMNISTDataset()
        {
            var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
        }
    }
}
