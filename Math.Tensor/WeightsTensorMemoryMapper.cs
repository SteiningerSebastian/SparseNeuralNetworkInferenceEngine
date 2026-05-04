using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Math.Tensor
{
    public class WeightsTensorMemoryMapper : ITensorMemoryLayout
    {
        protected int[] shape;
        protected int threads;
        protected int kernelLinesPerThread;

        public WeightsTensorMemoryMapper(int[] shape, int threads)
        {
            if (shape.Length != 2) throw new NotSupportedException("Only two dimensional tensors are supported.");

            this.shape = shape;
            this.threads = threads;

            this.kernelLinesPerThread = (int)MathF.Max(1, MathF.Ceiling(shape[0] / (Settings.KERNEL_SIZE / sizeof(float)) / (float) threads));
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

            const int floatsPerLine = (Settings.KERNEL_SIZE / sizeof(float));
            const int floatsPerKernel = floatsPerLine * (Settings.KERNEL_SIZE / sizeof(float));
            int cols = (int)MathF.Ceiling(shape[1] / (float)floatsPerLine);
            int floatsPerThread = floatsPerKernel * kernelLinesPerThread * cols;
            int threadId = index[0] / (floatsPerLine * kernelLinesPerThread);

            int colId = index[1] / floatsPerLine;

            int klt = kernelLinesPerThread;
            // The last thread may actually have fewer kernels so recalculate, to adjust. (Before we counted only full threads.)
            if (threadId == threads - 1)
            {
                // All the threads before the last one are actually full, the last one handles any remaining kernels.
                // So here it is caclaulated how many kernels there are in a coloumn and how many are remaining.
                // These are then used for the final in thread offset caclculation.
                klt = shape[0] / floatsPerLine - (threads-1) * kernelLinesPerThread;
            }

            int relativeRowId = index[0] % (floatsPerLine * klt);
            int relativeColId = index[1] % floatsPerLine;

            return threadId * floatsPerThread + colId * floatsPerKernel * klt + relativeRowId * floatsPerLine + relativeColId;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int[] MapToTensor(int offset)
        {
            const int floatsPerLine = (Settings.KERNEL_SIZE / sizeof(float));
            const int floatsPerKernel = floatsPerLine * (Settings.KERNEL_SIZE / sizeof(float));
            int cols = (int)MathF.Ceiling(shape[1] / (float)floatsPerLine);
            int floatsPerThread = floatsPerKernel * kernelLinesPerThread * cols;

            int threadId = offset / floatsPerThread;
            int threadRelativeOffset = offset - threadId * floatsPerThread;

            int klt = kernelLinesPerThread;
            // The last thread may actually have fewer kernels so recalculate, to adjust. (Before we counted only full threads.)
            if (threadId == threads - 1)
            {
                // All the threads before the last one are actually full, the last one handles any remaining kernels.
                // So here it is caclaulated how many kernels there are in a coloumn and how many are remaining.
                // These are then used for the final in thread offset caclculation.
                klt = shape[0] / floatsPerLine - (threads - 1) * kernelLinesPerThread;
            }

            int colId = threadRelativeOffset / (klt * floatsPerKernel);
            int colRelativeOffset = threadRelativeOffset - colId * floatsPerKernel * klt;
            int rowId = colRelativeOffset / floatsPerLine;
            int co = colRelativeOffset % floatsPerLine;

            return [threadId * kernelLinesPerThread * floatsPerLine + rowId, colId * floatsPerLine + co];
        }
    }
}
