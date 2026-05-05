using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.HardwareAcceleration;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;

namespace Meth.Tensor.Tests
{
    public class Tensor1DTests
    {
        [Fact]
        public void Constructor()
        {
            Tensor1D<float> tensor = new Tensor1D<float>(100, initialize: true);
            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(0, tensor[i]);
                tensor[i] = i;
            }

            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(i, tensor[i]);
            }
        }

        [Fact]
        public void Copy()
        {
            Tensor1D<float> tensor = new Tensor1D<float>(100, initialize: true);
            var tensorDC = tensor.DeepCopy(); // Create a deep copy
            var tensorC = (Tensor1D<float>)tensor.Clone();
            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(0, tensor[i]);
                tensor[i] = i;
            }

            for (int i = 0; i < 100; i++)
            {
                Assert.Equal(i, tensor[i]);
                Assert.Equal(i, tensorC[i]);
                Assert.Equal(0, tensorDC[i]);
            }
        }

        [Theory]
        [InlineData(16)]
        [InlineData(32)]
        [InlineData(64)]
        [InlineData(1024)]
        [InlineData(16*1024)]
        [InlineData(16*4096)]

        public async Task Add(int n)
        {
            IThreadPool pool = new SparseNeuralNetworkInferenceEngine.Engine.ThreadPool(1, 1024);
            IHardwareAccelerator accelerator = new AVXHardwareAccelerator(pool);

            IInferenceEngine engine = new InferenceEngine(accelerator);
            Tensor1D<float> a = engine.AllocateUninitializedAlignedTensor<Tensor1D<float>, float>(Enumerable.Range(0, n).Select(a => (float)a), n);
            Tensor1D<float> b = engine.AllocateUninitializedAlignedTensor<Tensor1D<float>, float>(Enumerable.Range(0, n).Select(a => (float)a), n);
            Tensor1D<float> c = engine.AllocateUninitializedAlignedTensor<Tensor1D<float>, float>(Enumerable.Range(0, n).Select(a => -(float)a), n);

            await a.AddAsync(b);

            for (int i = 0; i < n; i++)
            {
                Assert.Equal(i * 2, a[i]);
            }

            await a.AddAsync(c);

            for (int i = 0; i < n; i++)
            {
                Assert.Equal(i, a[i]);
            }
        }
    }
}