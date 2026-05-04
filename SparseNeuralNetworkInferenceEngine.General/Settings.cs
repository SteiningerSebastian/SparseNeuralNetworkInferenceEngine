using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public class Settings
    {
        /// <summary>
        /// The size of the cache lines.
        /// </summary>
        public const int CACHE_LINE_SIZE = 64;

        /// <summary>
        /// The size of the kernel in bytes. (Assuming 512bit registers are available.)
        /// </summary>
        public const int KERNEL_SIZE = 64; 
    }
}
