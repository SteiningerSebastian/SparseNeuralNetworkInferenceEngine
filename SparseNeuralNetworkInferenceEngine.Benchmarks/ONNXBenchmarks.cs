using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using Microsoft.VSDiagnostics;

using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using SparseNeuralNetworkInferenceEngine.General;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using System.Collections.Generic;
using System;

namespace SparseNeuralNetworkInferenceEngine.Benchmarks
{
    [CPUUsageDiagnoser]
    [MemoryDiagnoser]
    [HardwareCounters(HardwareCounter.BranchMispredictions, HardwareCounter.CacheMisses)]
    public class ONNXBenchmarks
    {
        const int BATCH_SIZE = 128;

        protected InferenceSession session;
        protected List<NamedOnnxValue> inputs;

        [GlobalSetup(Target = nameof(OnnxMNISTDataset))]
        public void SetupOnnxMNIST()
        {
            //Load data / load the first 128 samples of the MNIST test set, which is stored as a binary file of floats (flattened images)
            float[] data = BinaryLoader.ReadFileToFloatEnumerable("E:/IUBScSNNIE/Models/Evaluation/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 784).ToArray();

            session = new InferenceSession("E:/IUBScSNNIE/Models/Evaluation/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/model.onnx");

            int[] dimensions = { BATCH_SIZE, 784 };
            var inputTensor = new DenseTensor<float>(data, dimensions);

            string tfInputName = session.InputMetadata.Keys.First();
            inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor(tfInputName, inputTensor)
            };
        }

        [Benchmark]
        public void OnnxMNISTDataset()
        {
            var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
        }

        [GlobalSetup(Target = nameof(OnnxFashionMNISTDataset))]
        public void SetupOnnxFashionMNIST()
        {
            //Load data / load the first 128 samples of the MNIST test set, which is stored as a binary file of floats (flattened images)
            float[] data = BinaryLoader.ReadFileToFloatEnumerable("E:/IUBScSNNIE/Models/Evaluation/FASHION_MNIST_L_784_1024_1024_512_10_R_0e-0_ACC_88e-2_S_10e-2/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 784).ToArray();

            session = new InferenceSession("E:/IUBScSNNIE/Models/Evaluation/FASHION_MNIST_L_784_1024_1024_512_10_R_0e-0_ACC_88e-2_S_10e-2/model.onnx");

            int[] dimensions = { BATCH_SIZE, 784 };
            var inputTensor = new DenseTensor<float>(data, dimensions);

            string tfInputName = session.InputMetadata.Keys.First();
            inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor(tfInputName, inputTensor)
            };
        }

        [Benchmark]
        public void OnnxFashionMNISTDataset()
        {
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
        }
    }
}
