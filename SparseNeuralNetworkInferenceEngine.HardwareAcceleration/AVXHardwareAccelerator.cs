using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.HardwareAcceleration
{
    public class AVXHardwareAccelerator : IHardwareAccelerator, IAdd
    {
        public AVXHardwareAccelerator(IThreadPool threadPool)
        {

        }

        public void Add<T>(Span<T> addend1, Span<T> addend2) where T : INumber<T>
        {
            Debug.Assert(addend1.Length == addend2.Length); // Can only add elements of same length.

            unsafe
            {
                

            }
        }
    }
}
