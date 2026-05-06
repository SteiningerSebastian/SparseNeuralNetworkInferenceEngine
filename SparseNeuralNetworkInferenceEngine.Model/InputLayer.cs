using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public class InputLayer : ILayer
    {
        protected int[] shape;

        /// <summary>
        /// The first layer in the model defining the shape of the input.
        /// </summary>
        /// <param name="shape">The shape of the input tensor.</param>
        public InputLayer(int[] shape)
        {
            this.shape = shape;
        }

        /// <inheritdoc/>
        public int[] Compile(int[] inputShape, IInferenceEngine engine)
        {
            return shape;
        }

        /// <inheritdoc/>
        public async Task<Tensor<float>> InvokeAsync(Tensor<float> tensor, IInferenceEngine engine)
        {
            return tensor;
        }

        public void Load(IEnumerator<float> parameters)  {}

        public int NumerOfParameters() => 0;

        public int Store(Span<float> store) => 0;
    }
}