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
            float sum = 0;
            int i = 0;
            foreach (var v in values)
            {
                if (i >= length)
                    break;
                sum += System.MathF.Exp(v);
                i++;
            }

            i = 0;
            foreach (float v in values)
            {
                if (i >= length)
                    break;
                yield return System.MathF.Exp(v) / sum;
                i++;
            }
        }
    }
}
