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
        /// </summary>
        /// <typeparam name="T">Float or Double.</typeparam>
        /// <param name="addend1">The first slan of values.</param>
        /// <param name="addend2">The second span of values.</param>
        public Task AddAsync<T>(Span<T> addend1, Span<T> addend2) where T: unmanaged;
    }
}
