using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface ISparseFusedMultiplyAddReLU
    {
        /// <summary>
        /// Caclulates the result of ReLU(x W + b) and stores in activations.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="batches">The number of batches in the data.</param>
        /// <param name="weightsShape">The shape of the weights.</param>
        /// <param name="inputs">The inputs Tensor to use.</param>
        /// <param name="weights">The weights tensor.</param>
        /// <param name="bias">The bias tensor.</param>
        /// <param name="activations">The result is stored in the result Tensor.</param>
        public Task FusedMultiplyAdd(int batches, int[] weightsShape, Span<float> inputs, Span<float> weights, Span<float> bias, NativeMemoryOwner<float> activations, bool applyReLu = true, CancellationToken ct = default);
    }
}
