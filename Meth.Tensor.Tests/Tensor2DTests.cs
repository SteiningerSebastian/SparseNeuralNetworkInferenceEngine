using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using Newtonsoft.Json.Linq;

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
            var mapper = new BatchValueTensorMemoryLayout(shape);

            int offset = mapper.MapToMemory([b, a]);
            int[] index = mapper.MapToTensor(offset);

            Assert.Equal(b, index[0]);
            Assert.Equal(a, index[1]);
        }

        [Fact]
        public void Constructor()
        {
            var mapper = new BatchValueTensorMemoryLayout([64, 64]);
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

        [Fact]
        public void ConstructorRowMajor()
        {
            var mapper = new RowMajorTensorMemoryLayout([64, 64]);
            var values = Enumerable.Range(0, 64 * 64).Select(a => (float)a);
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

        [Fact]
        public void Enumerator()
        {
            var mapper = new BatchValueTensorMemoryLayout([64, 64]);
            Tensor2D<float> tensor = new Tensor2D<float>(64, 64, mapper, false, true, true, null);

            var values = Enumerable.Range(0, 64 * 64).Select(a => (float)a);
            tensor.PopulateWithEnumerable(values);

            tensor.SequenceEqual(values);

            Tensor2D<float> tensorZero = new Tensor2D<float>(128, 1024, mapper, true, true, true, null);
            Assert.Equal(0, tensorZero.Count(a => a != 0));
        }

        [Theory]
        [InlineData(128,128, 1)]
        [InlineData(128, 128, 2)]
        [InlineData(128, 128, 3)]
        [InlineData(128, 256, 3)]
        [InlineData(4, 128, 3)]
        [InlineData(128, 4, 3)]
        [InlineData(32, 32, 3)]
        [InlineData(289, 1024, 4)]
        [InlineData(264, 513, 8)]
        [InlineData(1027, 56, 7)]
        [InlineData(14, 256, 2)]
        [InlineData(4, 128, 5)]
        [InlineData(128, 4, 6)]
        [InlineData(32, 32, 2)]
        public void WeightsTensorMapperEnuemration(int d1, int d2, int p)
        {
            var mapper = new WeightsTensorMemoryMapper([d1, d2], p);
            var values = Enumerable.Range(0, d1 * d2).ToArray();

            var indexes = values.Select(a => mapper.MapToTensor(a)).ToArray();

            var restoredValues = indexes.Select(a=>mapper.MapToMemory(a)).ToArray();

            var restored = restoredValues.ToArray();

            for (int i = 0; i < restored.Length; i++)
            {
                if (restored[i] != values[i])
                {
                    Console.WriteLine("aaaa");
                }
            }
            

            Assert.True(restoredValues.SequenceEqual(values));
        }
    }
}
