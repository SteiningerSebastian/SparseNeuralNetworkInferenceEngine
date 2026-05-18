using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VSDiagnostics;
using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

namespace SparseNeuralNetworkInferenceEngine.Benchmarks
{
    [CPUUsageDiagnoser]
    [MemoryDiagnoser]
    [NativeMemoryProfiler]
    [HardwareCounters(HardwareCounter.BranchMispredictions, HardwareCounter.CacheMisses)]
    public class ONNXBenchmarks
    {
        const int BATCH_SIZE = 64;

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

        [GlobalSetup(Target = nameof(OnnxCifar10Dataset))]
        public void SetupOnnxCifar10()
        {
            //Load data / load the first 128 samples of the MNIST test set, which is stored as a binary file of floats (flattened images)
            float[] data = BinaryLoader.ReadFileToFloatEnumerable("E:/IUBScSNNIE/Models/Evaluation/CIFAR10_L_3072_4096_4096_512_10_R_0e-0_ACC_51e-2_AVG_S_68e-2/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 3 * 32 * 32).ToArray();

            session = new InferenceSession("E:/IUBScSNNIE/Models/Evaluation/CIFAR10_L_3072_4096_4096_512_10_R_0e-0_ACC_51e-2_AVG_S_68e-2/model.onnx");

            int[] dimensions = { BATCH_SIZE, 3 * 32 * 32 };
            var inputTensor = new DenseTensor<float>(data, dimensions);

            string tfInputName = session.InputMetadata.Keys.First();
            inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor(tfInputName, inputTensor)
            };
        }

        [Benchmark]
        public void OnnxCifar10Dataset()
        {
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();
        }
    }
}
