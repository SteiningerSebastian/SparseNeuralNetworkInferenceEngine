using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.HardwareAcceleration;
using SparseNeuralNetworkInferenceEngine.Math;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System.Diagnostics;
using ThreadPool = SparseNeuralNetworkInferenceEngine.Engine.ThreadPool;

namespace SparseNeuralNetworkInferenceEngine.Model.Tests
{
    public class TestModel
    {
        [Fact]
        public async Task SimpleModelCompilation()
        {
            Random random = new Random(0);

            IThreadPool threadPool = new ThreadPool(8, 1024);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
            IInferenceEngine engine = new InferenceEngine(accelerator);

            IModel model = new ModelSequential([
                new InputLayer([32, 784]),
                new DenseLayerAvx(304,threadPool.NumberOfThreads),
                new DenseLayerAvx(96, threadPool.NumberOfThreads),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            Tensor2D<float> inputs = engine.AllocateAlignedTensor<Tensor2D<float>, float>(new BatchValueTensorMemoryLayout([32, 784]), 32, 784);
            inputs.PopulateWithEnumerable(Enumerable.Range(0, 32 * 784).Select(a => random.NextSingle() * 2 - 1));

            var res = await model.InvokeAsync(inputs);
            Assert.NotNull(res);
        }

        [Fact]
        public async Task ModelLoadTest()
        {
            IThreadPool threadPool = new ThreadPool(Environment.ProcessorCount, 1024);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
            IInferenceEngine engine = new InferenceEngine(accelerator);

            const int BATCH_SIZE = 1;
            const int INPUT_SIZE = 784;

            ModelSequential model = new ModelSequential([
                new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(304,threadPool.NumberOfThreads),
                new DenseLayerAvx(112, threadPool.NumberOfThreads),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            var par = await BinaryLoader.ReadFileToFloatEnumerableAsync("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/MNIST304_112_10_Sparsity_59/model_parameters.bin")!;

            await model.LoadAsync("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/MNIST304_112_10_Sparsity_59/model_parameters.bin");

            float[] floats = model.Store();

            //for (int i = 0; i < floats.Length; i++)
            //{
            //    if(System.Math.Abs(floats[i] - par[i]) > 0.1)
            //    {
            //        Debug.WriteLine($"Parameters don't match, {floats[i]}, {par[i]} at {i}");
            //    }
            //}


            Assert.True(par.SequenceEqual(floats));
        }

        [Fact]
        public async Task BasicMNISTInferenceTest()
        {
            IThreadPool threadPool = new ThreadPool(Environment.ProcessorCount, 1024);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
            IInferenceEngine engine = new InferenceEngine(accelerator);

            const int BATCH_SIZE = 1;
            const int INPUT_SIZE = 784;

            ModelSequential model = new ModelSequential([
                new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(304,threadPool.NumberOfThreads),
                new DenseLayerAvx(112, threadPool.NumberOfThreads),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            await model.LoadAsync("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/MNIST304_112_10_Sparsity_59/model_parameters.bin");

            float[] inps = await BinaryLoader.ReadFileToFloatEnumerableAsync("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/MNIST304_112_10_Sparsity_59/x_test_flattened.bin");

            Tensor2D<float> inputs = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(new BatchValueTensorMemoryLayout([BATCH_SIZE, INPUT_SIZE]), BATCH_SIZE, INPUT_SIZE);
            inputs.PopulateWithEnumerable(inps.AsSpan().Slice(0, BATCH_SIZE * INPUT_SIZE).ToArray());

            var res = await model.InvokeAsync(inputs);

            float[] expected = await BinaryLoader.ReadFileToFloatEnumerableAsync("E:/IUBScSparseNeuralNetworkInferenceEngine/Models/MNIST304_112_10_Sparsity_59/y_test_predictions.bin");

            for (int batch = 0; batch < res.Shape[0]; batch++)
            {
                for (int i = 0; i < res.Shape[1]; i++)
                {
                    Assert.Equal(0, expected[BATCH_SIZE * batch + i] - res[batch, i], precision: 1);
                }
            }
        }
    }
}
