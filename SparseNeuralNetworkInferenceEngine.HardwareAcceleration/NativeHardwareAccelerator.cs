using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.HardwareAcceleration
{
    public class NativeHardwareAccelerator : IHardwareAccelerator
    {
        public object Clone()
        {
            return new NativeHardwareAccelerator();
        }
    }
}
