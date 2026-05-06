using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System.Diagnostics;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public class DenseLayerAvx : ILayer
    {
        protected Tensor2D<float>? weights;
        protected Tensor1D<float>? bias;
        protected Tensor2D<float>? activation;

        protected int size;
        protected int threadCount;
        protected bool useReLu;

        /// <summary>
        /// Creates a new Dense layer.
        /// </summary>
        /// <param name="size">The number of neurons in this layer.</param>
        public DenseLayerAvx(int size, int threadCount, bool useReLU = true)
        {
            if (size % (Settings.KERNEL_SIZE / sizeof(float)) != 0)
            {
                throw new NotSupportedException("Currently only layers with 16, 32, ..., x*16 neurons are supported.");
            }

            this.size = size;
            this.threadCount = threadCount;
            this.useReLu = useReLU;
        }

        /// <inheritdoc/>
        public int[] Compile(int[] inputShape, IInferenceEngine engine)
        {
            Debug.Assert(inputShape.Length == 2, $"Invalid shape ({string.Join(',', inputShape)}) for input tensor.");
            Debug.Assert(inputShape[1] % (Settings.KERNEL_SIZE / sizeof(float)) == 0, "Input-Shape must be a multiple of 16.");

            int inputLength = inputShape[1];
            int batchSize = inputShape[0];

            weights = engine.AllocateUninitializedPageAlignedTensor<Tensor2D<float>, float>(new WeightsTensorMemoryLayout([inputLength, size], threadCount), [inputLength, size]);
            bias = engine.AllocateUninitializedAlignedTensor<Tensor1D<float>, float>([size]);
            activation = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(new BatchValueTensorMemoryLayout([batchSize, size]), [batchSize, size]);

            return [batchSize, size];
        }


        /// <inheritdoc/>
        public async Task<Tensor<float>> InvokeAsync(Tensor<float> tensor, IInferenceEngine engine)
        {
            Debug.Assert(weights != null && bias != null && activation != null, "Can't invoke uncompiled layer.");
            Debug.Assert(tensor.GetType() == typeof(Tensor2D<float>), "Expected a Tensor2D of type float.");

            var input = (Tensor2D<float>)(object)tensor;

            await input.SparseFusedMultiplyAdd(weights, bias, activation);

            return activation;
        }

        public void Load(IEnumerator<float> parameters)
        {
            Debug.Assert(weights != null && bias != null && activation != null, "Can't load parameters of uncompiled model.");
            // Load the weights.
            for (int i = 0; i < weights.Shape[0]; i++)
            {
                for (int j = 0; j < weights.Shape[1]; j++)
                {
                    if (!parameters.MoveNext())
                        throw new IndexOutOfRangeException("Unable to load model from parameters.");
                    weights[i, j] = parameters.Current;
                }
            }

            // Load the bias.
            for (int i = 0; i < bias.Shape[0]; i++)
            {
                if (!parameters.MoveNext())
                    throw new IndexOutOfRangeException("Unable to load model from parameters.");
                bias[i] = parameters.Current;
            }
        }

        /// <inheritdoc/>
        public int NumerOfParameters()
        {
            return (weights?.Shape[0] ?? 0) * (weights?.Shape[1] ?? 0) + bias?.Length ?? 0;
        }

        /// <inheritdoc/>
        public int Store(Span<float> store)
        {
            Debug.Assert(weights != null && bias != null, "Unable to store model. (One or more parameter is null)");

            int offset = 0;
            for (int i = 0; i < weights.Shape[0]; i++)
            {
                for (int j = 0; j < weights.Shape[1]; j++)
                {
                    store[offset] = weights[i, j];
                    offset++;
                }
            }

            for (int b = 0; b < bias.Length; b++)
            {
                store[offset] = bias[b];
                offset++;
            }

            return offset;
        }
    }
}
