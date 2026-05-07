using Microsoft.AspNetCore.SignalR;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using SparseNeuralNetworkInferenceEngine.Model;

namespace SparseNeuralNetworkInferenceEngine.Example.MNISTWebserver
{
    public class MNISTHub : Hub
    {
        protected IModel model;
        protected IInferenceEngine engine;
        protected IDigitStore store;

        public MNISTHub(IModel model, IInferenceEngine engine, IDigitStore store)
        {
            this.model = model;
            this.engine = engine;
            this.store = store;
        }

        /// <summary>
        /// For a given normalized input image (represented as a float array of pixel values), this method runs the inference engine to predict the digit (0-9) that the image represents. The method returns the predicted digit as an integer.
        /// </summary>
        /// <param name="pixels">The row-major flattened array of pixel values representing the input image.</param>
        /// <returns>The predicted digit (0-9) as an integer.</returns>
        public async Task<int> GetNumber(float[] pixels)
        {
            var inputLayout = new BatchValueTensorMemoryLayout(1, 784);
            Tensor2D<float> inputs = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(inputLayout, 1, 784);
            inputs.PopulateWithEnumerable(pixels.AsSpan().Slice(0, 1 * 784).ToArray());

            var res = await model.InvokeAsync(inputs);

            // Get the index of the highest value in the result tensor, which corresponds to the predicted number.
            int predictedNumber = 0;
            for (int i = 1; i < res.Shape[1]; i++)
            {
                {
                    if (res.GetValue(0, i) > res.GetValue(0, predictedNumber))
                    {
                        predictedNumber = i;
                    }
                }
            }
            store.IncrementCounter(predictedNumber);

            await Clients.All.SendAsync("onincrement", store.DigitCoutner);

            return predictedNumber;
        }

        public async Task<int[]> ReceiveCounters()
        {
            return store.DigitCoutner;
        }
    }
}
