using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Math.Tensor
{
    public class WeightsTensorMemoryLayout : ITensorMemoryLayout
    {
        protected int[] shape;
        protected int threads;
        protected int vKernelsPerThread;
        protected int vKernelsRemaining;
        protected int vKernels;

        public WeightsTensorMemoryLayout(int[] shape, int threads)
        {
            if (shape.Length != 2) throw new NotSupportedException("Only two dimensional tensors are supported.");
            Debug.Assert(shape[0] % (Settings.KERNEL_SIZE / sizeof(float)) == 0, "Shape bust be a multiple of KERNEL_SIZE / 4");
            Debug.Assert(shape[1] % (Settings.KERNEL_SIZE / sizeof(float)) == 0, "Shape bust be a multiple of KERNEL_SIZE / 4");

            this.shape = shape;
            this.threads = threads;

            vKernels = shape[0] / (Settings.KERNEL_SIZE / sizeof(float));
            vKernelsPerThread = vKernels / threads;
            vKernelsRemaining = vKernels % threads;
        }

        public object Clone()
        {
            return new BatchValueTensorMemoryLayout(shape.ToArray());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int MapToMemory(int[] index)
        {
            Debug.Assert(index.Length == 2, "Unable to map index to memory offset. Only an two dimensional index is supported");

            const int KERNEL_SIZE_IN_FLOATS = (Settings.KERNEL_SIZE / sizeof(float));
            const int FLOATS_PER_KERNEL = KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
            int hKernels = shape[1] / KERNEL_SIZE_IN_FLOATS;

            // The first few are larger as they have the remaining kernel-lines distributed amonge them.
            // so check which thread is actually responsible.
            int row = index[0];
            int vKernel = row / KERNEL_SIZE_IN_FLOATS; // number of vertical kernels

            int kernelOffsetRemaining = vKernel - vKernelsRemaining * (vKernelsPerThread + 1);

            int rowThreadOffset;
            int threadOffset = 0;
            int vKernelsInThread;
           
            if (kernelOffsetRemaining < 0)
            {
                // All is in the larger partitons
                int threadId = vKernel / (vKernelsPerThread + 1);
                threadOffset = threadId * hKernels * (vKernelsPerThread + 1) * FLOATS_PER_KERNEL;
                vKernelsInThread = (vKernelsPerThread + 1);
                rowThreadOffset = threadId * (vKernelsPerThread + 1) * KERNEL_SIZE_IN_FLOATS;
            }
            else
            {
                // not everything in larger partitions / skip large partioons
                rowThreadOffset = vKernelsRemaining * (vKernelsPerThread + 1) * KERNEL_SIZE_IN_FLOATS;
                threadOffset = vKernelsRemaining * hKernels * (vKernelsPerThread + 1) * FLOATS_PER_KERNEL;
                int threadId = vKernelsRemaining + kernelOffsetRemaining / (vKernelsPerThread);
                threadOffset += threadId * hKernels * vKernelsPerThread * FLOATS_PER_KERNEL;
                vKernelsInThread = vKernelsPerThread;
                rowThreadOffset += threadId * vKernelsPerThread * KERNEL_SIZE_IN_FLOATS;
            }

            // Caclulate row countd from where the thread starts
            rowThreadOffset = row - rowThreadOffset;

            int rowOffset = rowThreadOffset * KERNEL_SIZE_IN_FLOATS;

            int colOffset = index[1] / KERNEL_SIZE_IN_FLOATS * vKernelsInThread * FLOATS_PER_KERNEL + index[1] % KERNEL_SIZE_IN_FLOATS;

            return threadOffset + rowOffset + colOffset;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int[] MapToTensor(int offset)
        {
            const int KERNEL_SIZE_IN_FLOATS = (Settings.KERNEL_SIZE / sizeof(float));
            const int FLOATS_PER_KERNEL = KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
            int hKernels = shape[1] / KERNEL_SIZE_IN_FLOATS;

            // Check in which thread we actually are
            int valuesInLargeThreads = (vKernelsPerThread + 1) * hKernels * FLOATS_PER_KERNEL;
            int valuesInNormalThread = vKernelsPerThread * hKernels * FLOATS_PER_KERNEL;
            int offsetAfterLargeThreads = offset - valuesInLargeThreads * vKernelsRemaining;

            // Offset within the thread
            int threadOffset = 0;
            int vKernelsInThread = vKernelsPerThread;
            int rowThreadStart = 0;
            if (offsetAfterLargeThreads < 0) //We are in the larger work-items
            {
                int threadId = offset / valuesInLargeThreads;
                threadOffset = offset - threadId * valuesInLargeThreads;
                vKernelsInThread = vKernelsPerThread + 1;
                rowThreadStart = threadId * (vKernelsInThread);
            }
            else //We are not in the large work items
            {
                rowThreadStart = vKernelsRemaining * (vKernelsPerThread + 1);
                threadOffset = offset - valuesInLargeThreads * vKernelsRemaining;
                int threadId = threadOffset / valuesInNormalThread;
                threadOffset -= valuesInNormalThread * (threadOffset / valuesInNormalThread);
                rowThreadStart += threadId * vKernelsInThread;
                threadId += vKernelsRemaining;
            }

            int valuesInColoumn = vKernelsInThread * FLOATS_PER_KERNEL;
            int colId = threadOffset / valuesInColoumn;
            int colOffset = threadOffset - colId * valuesInColoumn;
            int rowId = colOffset / KERNEL_SIZE_IN_FLOATS;
            int rowOffset = colOffset - rowId * KERNEL_SIZE_IN_FLOATS;

            return [rowThreadStart * KERNEL_SIZE_IN_FLOATS + rowId, colId * KERNEL_SIZE_IN_FLOATS + rowOffset];
        }
    }
}
