using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.HardwareAcceleration;
using SparseNeuralNetworkInferenceEngine.Math;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
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
                new DenseLayerAvx(256,threadPool.NumberOfThreads),
                new DenseLayerAvx(128, threadPool.NumberOfThreads),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
                ], engine);

            // Compile the model
            model.Compile();

            Tensor2D<float> inputs = engine.AllocateAlignedTensor<Tensor2D<float>, float>(32, 784);
            inputs.PopulateWithEnumerable(Enumerable.Range(0, 32*784).Select(a=>random.NextSingle()*2 -1));

            var res = await model.InvokeAsync(inputs);
            Assert.NotNull(res);
        }
    }
}
