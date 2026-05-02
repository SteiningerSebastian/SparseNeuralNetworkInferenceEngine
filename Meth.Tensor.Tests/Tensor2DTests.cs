using Math.Tensor;

namespace Meth.Tensor.Tests
{
    public class Tensor2DTests
    {
        [Theory]
        [InlineData(0,3)]
        [InlineData(0, 63)]
        [InlineData(13, 32)]
        [InlineData(17, 31)]
        [InlineData(51, 13)]
        [InlineData(61, 0)]
        public void BatchValueTensorMemoryMapper(int b, int a)
        {
            int[] shape = [64 , 64];
            var mapper = new BatchValueTensorMemoryMapper(shape);

            int offset = mapper.MapToMemory([b, a]);
            int[] index = mapper.MapToTensor(offset);

            Assert.Equal(b, index[0]);
            Assert.Equal(a, index[1]);
        }

        [Fact]
        public void Constructor()
        {
            var mapper = new BatchValueTensorMemoryMapper([64, 64]);
            var values = Enumerable.Range(0, 64*64).Select(a=>(float)a);
            Tensor2D<float> tensor = new Tensor2D<float>(64, 64, mapper, false, true, true, values);
            tensor.SequenceEqual(values);

            Tensor2D<float> tensorZero = new Tensor2D<float>(128, 1024, mapper, true, true, true, null);
            for (int i = 0; i < 128; i++)
            {
                for (int j = 0; j < 1024; j++)
                {
                    Assert.Equal(0, tensorZero[i, j]);
                }
            }
        }
    }
}
