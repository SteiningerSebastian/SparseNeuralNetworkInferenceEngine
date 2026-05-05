using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface IAddAligned
    {
        /// <summary>
        /// Adds all elements from addend1 and addend2 and stores the result in addend1.
        /// Requires the number of elements to be a multiple of KERNEL_SIZE_IN_FLOATS.
        /// </summary>
        /// <typeparam name="T">Float or Double.</typeparam>
        /// <param name="addend1">The first slan of values.</param>
        /// <param name="addend2">The second span of values.</param>
        public Task AddAsync(Span<float> addend1, Span<float> addend2);
    }
}
