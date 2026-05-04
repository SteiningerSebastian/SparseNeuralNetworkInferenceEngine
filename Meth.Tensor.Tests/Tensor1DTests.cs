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
    }
}