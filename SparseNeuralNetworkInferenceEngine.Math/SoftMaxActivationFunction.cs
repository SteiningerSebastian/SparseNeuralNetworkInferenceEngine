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

        public void Invoke(Tensor<float> values)
        {
            Debug.Assert(values.GetType() == typeof(Tensor2D<float>));

            var tensor = values as Tensor2D<float>;
            var shape = tensor!.Shape;

            unsafe
            {
                // Use stackalloc to create a temporary index array for accessing tensor elements
                // This avoids heap allocations and can improve performance when applying the softmax function.
                Span<int> index = stackalloc int[2];

                // Apply softmax over the last dimension
                for (int b = 0; b < shape[0]; b++)
                {
                    float sum = 0;
                    index[0] = b;
                    for (int i = 0; i < length; i++)
                    {
                        index[1] = i;
                        sum += System.MathF.Exp(tensor[index]);
                    }

                    for (int i = 0; i < length; i++)
                    {
                        index[1] = i;
                        tensor[index] = System.MathF.Exp(tensor[index]) / sum;
                    }
                }
            }
        }
    }
}
