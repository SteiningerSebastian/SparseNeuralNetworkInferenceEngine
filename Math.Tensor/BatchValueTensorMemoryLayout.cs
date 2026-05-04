using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Math.Tensor
{
    public class BatchValueTensorMemoryLayout : ITensorMemoryLayout
    {
        protected int[] shape;

        public BatchValueTensorMemoryLayout(int[] shape)
        {
            this.shape = shape;
        }

        public object Clone()
        {
            return new BatchValueTensorMemoryLayout(shape.ToArray());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int MapToMemory(int[] index)
        {
            // Removed by JIT Compiler in release
            Debug.Assert(index.Length == 2, "Unable to map index to memory offset. Only an two dimensional index is supported");

            const int l = (Settings.KERNEL_SIZE / sizeof(float));

            // Here we can't reduce because the integer division is needed. 
            return l * shape[0] * (index[1] / l) + index[0] * l + index[1] % l;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int[] MapToTensor(int offset)
        {
            int c = offset / ((Settings.KERNEL_SIZE / sizeof(float)) * shape[0]);
            int o = offset % ((Settings.KERNEL_SIZE / sizeof(float)) * shape[0]);
            int b = o / (Settings.KERNEL_SIZE / sizeof(float));
            int a = o % (Settings.KERNEL_SIZE / sizeof(float)) + c * (Settings.KERNEL_SIZE / sizeof(float));
            return [b, a];
        }
    }
}
