using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Diagnostics.Windows.Configs;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VSDiagnostics;
using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.HardwareAcceleration;
using SparseNeuralNetworkInferenceEngine.Math;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using SparseNeuralNetworkInferenceEngine.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;
using ThreadPool = SparseNeuralNetworkInferenceEngine.Engine.ThreadPool;

namespace SparseNeuralNetworkInferenceEngine.Benchmarks
{
    [CPUUsageDiagnoser]
    [MemoryDiagnoser]
    [NativeMemoryProfiler]
    [HardwareCounters(HardwareCounter.BranchMispredictions, HardwareCounter.CacheMisses)]
    public class SNNIEBenchmarks
    {
        const int BATCH_SIZE = 64;
        const int CORE_COUNT = 14;

        IModel model;
        Tensor2D<float> inputs;

        protected async Task LoadMNIST(string path)
        {
            IThreadPool threadPool = new ThreadPool(CORE_COUNT, 1024, System.Threading.ThreadPriority.Highest);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
            IInferenceEngine engine = new InferenceEngine(accelerator);

            const int INPUT_SIZE = 784;

            model = new ModelSequential([
                new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(304, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(112, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, accelerator, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            await model.LoadAsync($"E:/IUBScSNNIE/Models/Evaluation/{path}/model_parameters.bin");

            float[] inps = await BinaryLoader.ReadFileToFloatEnumerableAsync($"E:/IUBScSNNIE/Models/Evaluation/{path}/x_test_flattened.bin");

            var inputLayout = new BatchValueTensorMemoryLayout(BATCH_SIZE, INPUT_SIZE);
            inputs = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(inputLayout, BATCH_SIZE, INPUT_SIZE);
            inputs.PopulateWithEnumerable(inps.AsSpan().Slice(0, BATCH_SIZE * INPUT_SIZE).ToArray());
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineMNISTDataset))]
        public async Task SetupSparseNeuralNetworkInferenceEngineMNIST()
        {
            await LoadMNIST("MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineMNISTDataset()
        {
            await model.InvokeAsync(inputs);
        }

        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineMNISTDatasetS34))]
        public async Task SetupSparseNeuralNetworkInferenceEngineMNISTS34()
        {
            await LoadMNIST("MNIST_L_784_304_112_10_R_1e-5_ACC_97e-2_AVG_S_34e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineMNISTDatasetS34()
        {
            await model.InvokeAsync(inputs);
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineMNISTDatasetS63))]
        public async Task SetupSparseNeuralNetworkInferenceEngineMNISTS63()
        {
            await LoadMNIST("MNIST_L_784_304_112_10_R_5e-5_ACC_97e-2_AVG_S_63e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineMNISTDatasetS63()
        {
            await model.InvokeAsync(inputs);
        }


        protected async Task LoadFASHION_MNIST(string path)
        {
            IThreadPool threadPool = new ThreadPool(CORE_COUNT, 1024, System.Threading.ThreadPriority.Highest);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
            IInferenceEngine engine = new InferenceEngine(accelerator);

            const int INPUT_SIZE = 784;

            model = new ModelSequential([
                new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(1024, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(1024, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(512, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, accelerator, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            await model.LoadAsync($"E:/IUBScSNNIE/Models/Evaluation/{path}/model_parameters.bin");

            float[] inps = await BinaryLoader.ReadFileToFloatEnumerableAsync($"E:/IUBScSNNIE/Models/Evaluation/{path}/x_test_flattened.bin");

            var inputLayout = new BatchValueTensorMemoryLayout(BATCH_SIZE, INPUT_SIZE);
            inputs = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(inputLayout, BATCH_SIZE, INPUT_SIZE);
            inputs.PopulateWithEnumerable(inps.AsSpan().Slice(0, BATCH_SIZE * INPUT_SIZE).ToArray());
        }


        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineFASHION_MNISTDataset))]
        public async Task SetupSparseNeuralNetworkInferenceEngineFASHION_MNIST()
        {
            await LoadFASHION_MNIST("FASHION_MNIST_L_784_1024_1024_512_10_R_0e-0_ACC_88e-2_S_10e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineFASHION_MNISTDataset()
        {
            await model.InvokeAsync(inputs);
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS76))]
        public async Task SetupSparseNeuralNetworkInferenceEngineFASHION_MNISTS76()
        {
            await LoadFASHION_MNIST("FASHION_MNIST_L_784_1024_1024_512_10_R_1e-5_ACC_88e-2_S_76e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS76()
        {
            await model.InvokeAsync(inputs);
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS91))]
        public async Task SetupSparseNeuralNetworkInferenceEngineFASHION_MNISTS91()
        {
            await LoadFASHION_MNIST("FASHION_MNIST_L_784_1024_1024_512_10_R_5e-5_ACC_82e-2_S_91e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS91()
        {
            await model.InvokeAsync(inputs);
        }


        protected async Task Load_CIFAR10(string path)
        {
            IThreadPool threadPool = new ThreadPool(CORE_COUNT, 1024, System.Threading.ThreadPriority.Highest);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
            IInferenceEngine engine = new InferenceEngine(accelerator);

            const int INPUT_SIZE = 32*32*3;

            model = new ModelSequential([
                new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(4096, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(4096, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(512, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, accelerator, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            await model.LoadAsync($"E:/IUBScSNNIE/Models/Evaluation/{path}/model_parameters.bin");

            float[] inps = await BinaryLoader.ReadFileToFloatEnumerableAsync($"E:/IUBScSNNIE/Models/Evaluation/{path}/x_test_flattened.bin");

            var inputLayout = new BatchValueTensorMemoryLayout(BATCH_SIZE, INPUT_SIZE);
            inputs = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(inputLayout, BATCH_SIZE, INPUT_SIZE);
            inputs.PopulateWithEnumerable(inps.AsSpan().Slice(0, BATCH_SIZE * INPUT_SIZE).ToArray());
        }


        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineCIFAR10Dataset))]
        public async Task SetupSparseNeuralNetworkInferenceEngineCIFAR10()
        {
            await Load_CIFAR10("CIFAR10_L_3072_4096_4096_512_10_R_0e-0_ACC_51e-2_AVG_S_68e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineCIFAR10Dataset()
        {
            await model.InvokeAsync(inputs);
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineCIFAR10DatasetS82))]
        public async Task SetupSparseNeuralNetworkInferenceEngineCIFAR10S82()
        {
            await Load_CIFAR10("CIFAR10_L_3072_4096_4096_512_10_R_1e-6_ACC_44e-2_AVG_S_82e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineCIFAR10DatasetS82()
        {
            await model.InvokeAsync(inputs);
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineCIFAR10DatasetS91))]
        public async Task SetupSparseNeuralNetworkInferenceEngineCIFAR10S91()
        {
            await Load_CIFAR10("CIFAR10_L_3072_4096_4096_512_10_R_25e-7_ACC_41e-2_AVG_S_91e-2");
        }

        [Benchmark]
        public async Task SparseneuralNetworkInferenceEngineCIFAR10DatasetS91()
        {
            await model.InvokeAsync(inputs);
        }
    }
}
