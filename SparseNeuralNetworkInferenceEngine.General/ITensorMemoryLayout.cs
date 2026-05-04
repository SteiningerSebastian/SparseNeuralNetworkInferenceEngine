using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface ITensorMemoryLayout: ICloneable
    {
        /// <summary>
        /// Maps a given index to a given memory position. (Bijection of index to offset)
        /// </summary>
        /// <param name="index">The index in the tensor.</param>
        /// <returns>The offset of the value.</returns>
        public int MapToMemory(int[] index);

        /// <summary>
        /// Map the offset to a index.
        /// </summary>
        /// <param name="offset">The memory offset. </param>
        /// <returns>The corresponding index is returned.</returns>
        public int[] MapToTensor(int offset);
    }
}
