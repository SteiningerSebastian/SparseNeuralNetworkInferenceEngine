using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Math.Tensor
{
    public class RowMajorTensorMemoryLayout : ITensorMemoryLayout
    {
        protected int[] shape;
        protected int[] elementsPerDimension;

        /// <summary>
        /// Creates a new mapper that for 1D tensors maps identity for others maps dimensions in row major.
        /// </summary>
        /// <param name="shape">The shape of the tensor for which the mapper should be defined.</param>
        public RowMajorTensorMemoryLayout(int[] shape){
            this.shape = shape;
            this.elementsPerDimension = new int[shape.Length];
            elementsPerDimension[shape.Length-1] = 1;

            for (int s = shape.Length - 2; s >= 0; s--)
            {
                elementsPerDimension[s] = shape[s+1] * elementsPerDimension[s+1];
            }
        }

        public object Clone()
        {
            return new RowMajorTensorMemoryLayout(shape.ToArray());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int MapToMemory(int[] index)
        {
            Debug.Assert(index.Length == shape.Length, $"Identity wrapper expects an indes of shape ({string.Join(',', shape)}) but got an index of shape ({string.Join(',', index)}).");
    
            int offset = 0;

            for(int s = 0; s < shape.Length; s++)
            {
                offset += elementsPerDimension[s] * index[s];
            }

            return offset;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int[] MapToTensor(int offset)
        {
            int[] index = new int[shape.Length];

            for (int s = 0; s < shape.Length; s++)
            {
                index[s] = offset / elementsPerDimension[s];
                offset = offset - index[s] * (offset / elementsPerDimension[s]);
            }

            return index;
        }
    }
}
