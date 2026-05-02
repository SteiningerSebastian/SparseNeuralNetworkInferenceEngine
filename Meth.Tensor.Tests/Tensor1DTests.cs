using Math.Tensor;

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
        [InlineData(100)]
        [InlineData(1000)]
        [InlineData(10000)]
        [InlineData(10001)]
        [InlineData(10002)]
        [InlineData(10003)]
        [InlineData(10004)]
        [InlineData(10005)]
        [InlineData(10006)]
        [InlineData(10007)]
        [InlineData(10008)]
        [InlineData(10009)]
        [InlineData(10010)]
        [InlineData(10011)]
        [InlineData(10012)]
        [InlineData(10013)]
        [InlineData(10014)]
        [InlineData(10015)]
        [InlineData(10016)]
        public void Addition(int size)
        {
            Tensor1D<float> tensor = new Tensor1D<float>(size);
            for (int i = 0; i < size; i++)
            {
                tensor[i] = i;
            }

            Tensor1D<float> tensorB = (Tensor1D<float>)tensor.DeepCopy(); // Creating an equal tensor

            tensor.Add(tensorB);

            for (int i = 0; i < size; i++)
            {
                Assert.Equal(2*i, tensor[i]);
            }
        }
    }
}
