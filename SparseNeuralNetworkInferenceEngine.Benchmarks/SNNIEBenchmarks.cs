using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
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
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;

namespace SparseNeuralNetworkInferenceEngine.Benchmarks
{
    [CPUUsageDiagnoser]
    [MemoryDiagnoser]
    [HardwareCounters(HardwareCounter.BranchMispredictions, HardwareCounter.CacheMisses)]
    public class SNNIEBenchmarks
    {
        const int BATCH_SIZE = 128;

        IModel model;
        Tensor2D<float> inputs;

        protected async Task LoadMNIST(string path)
        {
            IThreadPool threadPool = new ThreadPool(Environment.ProcessorCount, 1024, System.Threading.ThreadPriority.Highest);
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
        public void SparseneuralNetworkInferenceEngineMNISTDataset()
        {
            model.InvokeAsync(inputs).Wait();
        }

        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineMNISTDatasetS34))]
        public async Task SetupSparseNeuralNetworkInferenceEngineMNISTS34()
        {
            await LoadMNIST("MNIST_L_784_304_112_10_R_1e-5_ACC_97e-2_AVG_S_34e-2");
        }

        [Benchmark]
        public void SparseneuralNetworkInferenceEngineMNISTDatasetS34()
        {
            model.InvokeAsync(inputs).Wait();
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineMNISTDatasetS63))]
        public async Task SetupSparseNeuralNetworkInferenceEngineMNISTS63()
        {
            await LoadMNIST("MNIST_L_784_304_112_10_R_5e-5_ACC_97e-2_AVG_S_63e-2");
        }

        [Benchmark]
        public void SparseneuralNetworkInferenceEngineMNISTDatasetS63()
        {
            model.InvokeAsync(inputs).Wait();
        }


        protected async Task LoadFASHION_MNIST(string path)
        {
            IThreadPool threadPool = new ThreadPool(16, 1024, System.Threading.ThreadPriority.Highest);
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
        public void SparseneuralNetworkInferenceEngineFASHION_MNISTDataset()
        {
            model.InvokeAsync(inputs).Wait();
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS76))]
        public async Task SetupSparseNeuralNetworkInferenceEngineFASHION_MNISTS76()
        {
            await LoadFASHION_MNIST("FASHION_MNIST_L_784_1024_1024_512_10_R_1e-5_ACC_88e-2_S_76e-2");
        }

        [Benchmark]
        public void SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS76()
        {
            model.InvokeAsync(inputs).Wait();
        }



        [GlobalSetup(Target = nameof(SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS91))]
        public async Task SetupSparseNeuralNetworkInferenceEngineFASHION_MNISTS91()
        {
            await LoadFASHION_MNIST("FASHION_MNIST_L_784_1024_1024_512_10_R_5e-5_ACC_82e-2_S_91e-2");
        }

        [Benchmark]
        public void SparseneuralNetworkInferenceEngineFASHION_MNISTDatasetS91()
        {
            model.InvokeAsync(inputs).Wait();
        }
    }
}
