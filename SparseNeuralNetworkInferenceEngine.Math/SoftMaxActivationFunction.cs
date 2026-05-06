using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Math
{
    public class SoftMaxActivationFunction : IActivationFunction
    {
        protected int length;
        /// <summary>
        /// Creates a softmax activation function that is applied over a given number of inputs.
        /// </summary>
        /// <param name="length">The length to apply the function to (-1 for infinite)</param>
        public SoftMaxActivationFunction(int length = -1)
        {
            this.length = length;
        }

        public IEnumerable<float> Invoke(IEnumerable<float> values)
        {
            float sum = values.Take(length).Select(v=>System.MathF.Exp(v)).Sum();
            return values.Select(v=>System.MathF.Exp(v) /  sum);
        }
    }
}
