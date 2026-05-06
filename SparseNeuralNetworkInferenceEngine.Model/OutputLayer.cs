using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public class OutputLayer : ILayer
    {
        protected int length;
        /// <summary>
        /// Creates the output layer.
        /// </summary>
        /// <param name="length">The number of neurons in the output layer. (dynamic casted to)</param>
        public OutputLayer(int length)
        {
            this.length = length;
        }

        public int[] Compile(int[] inputShape, IInferenceEngine engine)
        {
            return [inputShape[0], length];
        }

        public async Task<Tensor<float>> InvokeAsync(Tensor<float> tensor, IInferenceEngine engine)
        {
            return tensor.DynamicCast([tensor.Shape[0], length]);
        }
    }
}
