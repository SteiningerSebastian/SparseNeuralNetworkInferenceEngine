using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.HardwareAcceleration
{
    public class HardwareAccelerationException : Exception
    {
        public HardwareAccelerationException(string details) : base($"Unable to use hardware acceleration: {details} ")
        {
        }
    }
}
