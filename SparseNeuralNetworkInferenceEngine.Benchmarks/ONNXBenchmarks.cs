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
        const string MODEL_DIR = "E:/IUBScSNNIE/Models/Evaluation";

        protected InferenceSession session;
        protected List<NamedOnnxValue> inputs;

        protected static SessionOptions GetSessionOptions()
        {
            // These are here to make the ONNX benchmarks more comparable to the custom engine benchmarks, which use 6 threads and have thread affinities set to the first 6 cores.
            // You can adjust these settings as needed for your specific hardware and use case. (only 6 P-Cores on the ultra 9 285H)
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.IntraOpNumThreads = 6;
            sessionOptions.AddSessionConfigEntry("session.intra_op_thread_affinities", "1;2;3;4;5");
            return sessionOptions;
        }

        [GlobalSetup(Target = nameof(OnnxMNISTDataset))]
        public void SetupOnnxMNIST()
        {
            //Load data / load the first 128 samples of the MNIST test set, which is stored as a binary file of floats (flattened images)
            float[] data = BinaryLoader.ReadFileToFloatEnumerable($"{MODEL_DIR}/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 784).ToArray();

            session = new InferenceSession($"{MODEL_DIR}/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/model.onnx", GetSessionOptions());

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
            float[] data = BinaryLoader.ReadFileToFloatEnumerable($"{MODEL_DIR}/FASHION_MNIST_L_784_1024_1024_512_10_R_0e-0_ACC_88e-2_S_10e-2/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 784).ToArray();

            session = new InferenceSession($"{MODEL_DIR}/FASHION_MNIST_L_784_1024_1024_512_10_R_0e-0_ACC_88e-2_S_10e-2/model.onnx", GetSessionOptions());

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
            float[] data = BinaryLoader.ReadFileToFloatEnumerable($"{MODEL_DIR}/CIFAR10_L_3072_4096_4096_512_10_R_0e-0_ACC_51e-2_AVG_S_68e-2/x_test_flattened.bin");
            data = data.Take(BATCH_SIZE * 3 * 32 * 32).ToArray();

            session = new InferenceSession($"{MODEL_DIR}/CIFAR10_L_3072_4096_4096_512_10_R_0e-0_ACC_51e-2_AVG_S_68e-2/model.onnx", GetSessionOptions());

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
