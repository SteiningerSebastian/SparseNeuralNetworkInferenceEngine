using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface ITensor<K>
    {
        /// <summary>
        /// The shape of the tensor.
        /// </summary>
        public int[] Shape { get; }
    }
}
