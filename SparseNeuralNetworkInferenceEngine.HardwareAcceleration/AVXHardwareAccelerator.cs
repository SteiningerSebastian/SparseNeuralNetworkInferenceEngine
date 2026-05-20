using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SparseNeuralNetworkInferenceEngine.HardwareAcceleration
{
    public class AVXHardwareAccelerator : IHardwareAccelerator, ISparseFusedMultiplyAddReLU, IDisposable
    {
        protected IThreadPool threadPool;
        protected NativeMemoryBufferManager<float> bufferManager;
        protected NativeMemoryOwner<SFMAWorkItem> workItemBuffer;
        protected List<Task> tasks;
        protected NativeMemoryBuffer<float>? threadedResults;
        protected NativeMemoryOwner<int> threadedResultsArrivedCount;


        [StructLayout(LayoutKind.Explicit, Size = 128)]
        protected unsafe struct SFMAWorkItem
        {
            [FieldOffset(0)] public float* currentInputsPtr;
            [FieldOffset(8)] public float* currentBufferPtr;
            [FieldOffset(16)] public float* currentWeightsPtr;
            [FieldOffset(24)] public float* ptrRawBuffer;
            [FieldOffset(32)] public float* ptrBias;
            [FieldOffset(40)] public float* ptrActivations;

            [FieldOffset(48)] public int hKernels;
            [FieldOffset(52)] public int vHeightInPartition;
            [FieldOffset(56)] public int batches;
            [FieldOffset(60)] public int partitionId;
            [FieldOffset(64)] public int partitions;

            [FieldOffset(76)] public bool applyReLU;

            [FieldOffset(77)] public int* arrivedThreadCount;

            // Remaining bytes are padding.
        }

        public AVXHardwareAccelerator(IThreadPool threadPool)
        {
            this.threadPool = threadPool;

            bufferManager = new NativeMemoryBufferManager<float>(-1);

            if (Vector<float>.Count <= 4)
            {
                throw new HardwareAccelerationException("This hardware accelerator requieres support for 256bit or 512bit Avx / SIMD!");
            }

            // Preallocate the buffers for the inter-thread communication.

            workItemBuffer = new NativeMemoryOwner<SFMAWorkItem>((nuint)threadPool.NumberOfThreads, (nuint)128, 64);

            tasks = new List<Task>(threadPool.NumberOfThreads);

            // createing the arrived count and allign it to a cache line to avoid false sharing.
            threadedResultsArrivedCount = new NativeMemoryOwner<int>(1, (nuint)sizeof(int), 64);
        }

        public object Clone()
        {
            return new AVXHardwareAccelerator(threadPool);
        }

        public void PrepareForInference(int[] shape)
        {
            // Nothing to prepare for inference if the shape is empty.
            if (shape.Length == 0)
                return;

            int length = 1;
            foreach (int dim in shape)
            {
                length *= dim;
            }

            // For each thread we need a buffer of the size of the activations to cache the intermediary results.
            length *= threadPool.NumberOfThreads;

            // Preallocate a buffer of the required size to avoid overhead during inference.
            bufferManager.GetBuffer(length, sizeof(float));
        }

        /// <summary>
        /// While this function is basically the same as the one above, to avoid branching inside such high performance code
        /// we created this function again with the only difference that Max(0, element) is applied before storing in addend1
        /// </summary>
        protected Task AddReLUAsync(Span<float> addend1, Span<float> addend2)
        {
            throw new NotImplementedException();
        }


        const int KERNEL_SIZE_IN_FLOATS = (Settings.KERNEL_SIZE / sizeof(float));

        public unsafe static void FusedMultiplyAddReLUWorkSequential(int threadId, void* data)
        {
            SFMAWorkItem item = (*(SFMAWorkItem*)data);
            float* currentInputsPtr = item.currentInputsPtr;
            float* currentBufferPtr = item.currentBufferPtr;
            float* currentWeightsPtr = item.currentWeightsPtr;
            float* ptrRawBuffer = item.ptrRawBuffer;
            float* ptrBias = item.ptrBias;
            float* ptrActivations = item.ptrActivations;

            int hKernels = item.hKernels;
            int vHeightInPartition = item.vHeightInPartition;
            int batches = item.batches;
            int partitionId = item.partitionId;
            int partitions = item.partitions;

            int hKernelsPerPartition = hKernels / item.partitions;
            int hKernelsRemaining = hKernels % item.partitions;

            bool applyReLU = item.applyReLU;

            float* coloumnStartInputsPtr = currentInputsPtr;

            for (int c = 0; c < hKernels; c++)
            {
                float* coloumnStartBufferPtr = currentBufferPtr;
                currentInputsPtr = coloumnStartInputsPtr;

                // Unrolling the first iteration of the loop.
                {
                    // these rows are in the same part of the buffer
                    currentBufferPtr = coloumnStartBufferPtr;
                    float* kernelStartWeightsPtr = currentWeightsPtr;

                    // This kernel for every batch to keep weights in cahce
                    for (int b = 0; b < batches; b++)
                    {
                        // For the batch start at the beginning of the weights (weights and inputs)
                        currentWeightsPtr = kernelStartWeightsPtr;

                        var vcInputs1 = Vector.LoadAligned(currentInputsPtr);
                        var vcInputs2 = Vector.LoadAligned(currentInputsPtr + Vector<float>.Count);

                        // Check if whole kernel is zero => skip sparse activations.
                        if (!(Vector.EqualsAll(Vector<float>.Zero, vcInputs1) && Vector.EqualsAll(Vector<float>.Zero, vcInputs2)))
                        {
                            Vector<float> addents1 = Vector<float>.Zero;
                            Vector<float> addents2 = Vector<float>.Zero;

                            for (int ri = 0; ri < KERNEL_SIZE_IN_FLOATS; ri++)
                            {
                                Vector<float> vecWeights1 = Vector.LoadAligned(currentWeightsPtr);
                                Vector<float> vecWeights2 = Vector.LoadAligned(currentWeightsPtr + Vector<float>.Count);

                                float x = *currentInputsPtr; // Load the input from inputs
                                addents1 = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights1, addents1);
                                addents2 = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights2, addents2);

                                // move the weigths point to next line.
                                currentWeightsPtr += KERNEL_SIZE_IN_FLOATS;

                                currentInputsPtr += 1; // Increas inputs pointer for next line
                            }

                            Vector.Store(addents1, currentBufferPtr); // Store the result back in the buffer.
                            Vector.Store(addents2, currentBufferPtr + Vector<float>.Count); // Store the result back in the buffer.
                        }
                        else
                        {
                            Vector.Store(Vector<float>.Zero, currentBufferPtr);
                            Vector.Store(Vector<float>.Zero, currentBufferPtr + Vector<float>.Count);

                            currentWeightsPtr += KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                            currentInputsPtr += KERNEL_SIZE_IN_FLOATS;
                        }

                        // Move the buffer we work on.
                        currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                    }
                }

                // The amount of rows we need to compute before moving on the the next coloumn
                for (int r = 1; r < vHeightInPartition; r += 1)
                {
                    // these rows are in the same part of the buffer
                    currentBufferPtr = coloumnStartBufferPtr;

                    float* kernelStartWeightsPtr = currentWeightsPtr;

                    // This kernel for every batch to keep weights in cahce
                    for (int b = 0; b < batches; b++)
                    {
                        // For the batch start at the beginning of the weights (weights and inputs)
                        currentWeightsPtr = kernelStartWeightsPtr;

                        var vcInputs1 = Vector.LoadAligned(currentInputsPtr);
                        var vcInputs2 = Vector.LoadAligned(currentInputsPtr + Vector<float>.Count);

                        //Check if whole kernel is zero => skip sparse activations.
                        if (!(Vector.EqualsAll(Vector<float>.Zero, vcInputs1) && Vector.EqualsAll(Vector<float>.Zero, vcInputs2)))
                        {
                            Vector<float> addents1 = Vector.LoadAligned(currentBufferPtr);
                            Vector<float> addents2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);

                            for (int ri = 0; ri < KERNEL_SIZE_IN_FLOATS; ri++)
                            {
                                float x = *currentInputsPtr; // Load the input from inputs

                                Vector<float> vecWeights1 = Vector.LoadAligned(currentWeightsPtr);
                                addents1 = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights1, addents1);

                                Vector<float> vecWeights2 = Vector.LoadAligned(currentWeightsPtr + Vector<float>.Count);
                                addents2 = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights2, addents2);

                                // move the weigths point to next line.
                                currentWeightsPtr += KERNEL_SIZE_IN_FLOATS;

                                currentInputsPtr += 1; // Increas inputs pointer for next line
                            }

                            Vector.Store(addents1, currentBufferPtr); // Store the result back in the buffer.
                            Vector.Store(addents2, currentBufferPtr + Vector<float>.Count); // Store the result back in the buffer.
                        }
                        else
                        {
                            currentWeightsPtr += KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                            currentInputsPtr += KERNEL_SIZE_IN_FLOATS;
                        }

                        // Move the buffer we work on.
                        currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                    }
                }
            }

            // Signal that this thread has finished the multiplication and wait for the others to finish as well.
            if (Interlocked.Increment(ref *item.arrivedThreadCount) != partitions)
            { // only if we are not the last thread to arive, we wait
               
                while (Volatile.Read(ref *item.arrivedThreadCount) != partitions)
                {
                    // Here we are on the very hot path of the code, so we want to avoid yielding the thread if possible.
                    Thread.SpinWait(1);
                }
            }

            int cols = hKernelsPerPartition;
            // Distribute the remaining coloumns to the first few partitions.
            if (partitionId < hKernelsRemaining)
                cols++;

            int offset = partitionId * hKernelsPerPartition * KERNEL_SIZE_IN_FLOATS +
                         (partitionId < hKernelsRemaining ? partitionId : hKernelsRemaining) * KERNEL_SIZE_IN_FLOATS;

            currentBufferPtr = ptrRawBuffer + offset * batches;

            // Offset between intermediary results in the buffer
            int bufferOffset = batches * hKernels * KERNEL_SIZE_IN_FLOATS;

            float* startBiasPtr = ptrBias + offset;

            //If there are no coloumns to process we can skip this step and let the other threads do their work.
            if (cols != 0)
            {
                float* startActivationPtr = ptrActivations + offset * batches;
                #region LoopSum

                {   // Unrolling the loop to get rid of the branch.
                    float* startBufferPtr = currentBufferPtr;
                    float* currentActivationPtr = startActivationPtr;
                    float* currentBiasPtr = startBiasPtr;

                    if (applyReLU && partitions - 1 == 0)
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent1 = Vector.LoadAligned(currentBiasPtr);
                                Vector<float> buf1 = Vector.LoadAligned(currentBufferPtr);
                                var res1 = Vector.Add(buf1, addent1);
                                res1 = Vector.Max(res1, Vector<float>.Zero);
                                res1.StoreAligned(currentActivationPtr);

                                Vector<float> addent2 = Vector.LoadAligned(currentBiasPtr + Vector<float>.Count);
                                Vector<float> buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                var res2 = Vector.Add(buf2, addent2);
                                res2 = Vector.Max(res2, Vector<float>.Zero);
                                res2.StoreAligned(currentActivationPtr + Vector<float>.Count);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    else
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent1 = Vector.LoadAligned(currentBiasPtr);
                                Vector<float> buf1 = Vector.LoadAligned(currentBufferPtr);
                                var res1 = Vector.Add(buf1, addent1);
                                res1.StoreAligned(currentActivationPtr);

                                Vector<float> addent2 = Vector.LoadAligned(currentBiasPtr + Vector<float>.Count);
                                Vector<float> buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                var res2 = Vector.Add(buf2, addent2);
                                res2.StoreAligned(currentActivationPtr + Vector<float>.Count);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.

                }

                // For the result of each partition add the intermediar results togehter.
                for (int p = 1; p < partitions - 1; p++)
                {
                    float* startBufferPtr = currentBufferPtr;
                    float* currentActivationPtr = startActivationPtr;
                    float* currentBiasPtr = startBiasPtr;

                    for (int c = 0; c < cols; c++)
                    {
                        for (int b = 0; b < batches; b++)
                        {
                            Vector<float> addent1 = Vector.LoadAligned(currentActivationPtr);
                            Vector<float> buf1 = Vector.LoadAligned(currentBufferPtr);
                            var res1 = Vector.Add(buf1, addent1);
                            res1.StoreAligned(currentActivationPtr);

                            Vector<float> addent2 = Vector.LoadAligned(currentActivationPtr + Vector<float>.Count);
                            Vector<float> buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                            var res2 = Vector.Add(buf2, addent2);
                            res2.StoreAligned(currentActivationPtr + Vector<float>.Count);

                            currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                            currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                        currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                    }
                    currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.
                }

                // The end of the loop
                if (partitions >= 2)
                {
                    float* startBufferPtr = currentBufferPtr;
                    float* currentActivationPtr = startActivationPtr;
                    float* currentBiasPtr = startBiasPtr;

                    if (applyReLU)
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent1 = Vector.LoadAligned(currentActivationPtr);
                                Vector<float> buf1 = Vector.LoadAligned(currentBufferPtr);
                                var res1 = Vector.Add(buf1, addent1);
                                res1 = Vector.Max(res1, Vector<float>.Zero);
                                res1.StoreAligned(currentActivationPtr);

                                Vector<float> addent2 = Vector.LoadAligned(currentActivationPtr + Vector<float>.Count);
                                Vector<float> buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                var res2 = Vector.Add(buf2, addent2);
                                res2 = Vector.Max(res2, Vector<float>.Zero);
                                res2.StoreAligned(currentActivationPtr + Vector<float>.Count);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    else
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent1 = Vector.LoadAligned(currentActivationPtr);
                                Vector<float> buf1 = Vector.LoadAligned(currentBufferPtr);
                                var res1 = Vector.Add(buf1, addent1);
                                res1.StoreAligned(currentActivationPtr);

                                Vector<float> addent2 = Vector.LoadAligned(currentActivationPtr + Vector<float>.Count);
                                Vector<float> buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                var res2 = Vector.Add(buf2, addent2);
                                res2.StoreAligned(currentActivationPtr + Vector<float>.Count);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.
                }
                #endregion
            }
        }

        public unsafe static void FusedMultiplyAddReLUWorkNative(int threadId, void* data)
        {
            SFMAWorkItem item = (*(SFMAWorkItem*)data);
            float* currentInputsPtr = item.currentInputsPtr;
            float* currentBufferPtr = item.currentBufferPtr;
            float* currentWeightsPtr = item.currentWeightsPtr;
            float* ptrRawBuffer = item.ptrRawBuffer;
            float* ptrBias = item.ptrBias;
            float* ptrActivations = item.ptrActivations;

            int hKernels = item.hKernels;
            int vHeightInPartition = item.vHeightInPartition;
            int batches = item.batches;
            int partitionId = item.partitionId;
            int partitions = item.partitions;

            int hKernelsPerPartition = hKernels / item.partitions;
            int hKernelsRemaining = hKernels % item.partitions;

            bool applyReLU = item.applyReLU;

            float* coloumnStartInputsPtr = currentInputsPtr;

            for (int c = 0; c < hKernels; c++)
            {
                float* coloumnStartBufferPtr = currentBufferPtr;
                currentInputsPtr = coloumnStartInputsPtr;

                // The first iteration unrolled.
                {
                    // these rows are in the same part of the buffer
                    currentBufferPtr = coloumnStartBufferPtr;
                    float* kernelStartWeightsPtr = currentWeightsPtr;

                    // This kernel for every batch to keep weights in cahce
                    for (int b = 0; b < batches; b++)
                    {
                        // For the batch start at the beginning of the weights (weights and inputs)
                        currentWeightsPtr = kernelStartWeightsPtr;


                        var vcInputs = Vector.LoadAligned(currentInputsPtr);

                        // Check if whole kernel is zero => skip sparse activations.
                        if (!Vector.EqualsAll(Vector<float>.Zero, vcInputs))
                        {
                            Vector<float> addents = Vector<float>.Zero;

                            for (int ri = 0; ri < KERNEL_SIZE_IN_FLOATS; ri++)
                            {
                                Vector<float> vecWeights = Vector.LoadAligned(currentWeightsPtr);

                                float x = *currentInputsPtr; // Load the input from inputs
                                addents = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights, addents);

                                // move the weigths point to next line.
                                currentWeightsPtr += KERNEL_SIZE_IN_FLOATS;

                                currentInputsPtr += 1; // Increas inputs pointer for next line
                            }

                            Vector.Store(addents, currentBufferPtr); // Store the result back in the buffer.
                        }
                        else
                        {
                            Vector.Store(Vector<float>.Zero, currentBufferPtr);

                            currentWeightsPtr += KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                            currentInputsPtr += KERNEL_SIZE_IN_FLOATS;
                        }

                        // Move the buffer we work on.
                        currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                    }
                }


                // The amount of rows we need to compute before moving on the the next coloumn
                for (int r = 1; r < vHeightInPartition; r += 1)
                {
                    // these rows are in the same part of the buffer
                    currentBufferPtr = coloumnStartBufferPtr;
                    float* kernelStartWeightsPtr = currentWeightsPtr;

                    // This kernel for every batch to keep weights in cahce
                    for (int b = 0; b < batches; b++)
                    {
                        // For the batch start at the beginning of the weights (weights and inputs)
                        currentWeightsPtr = kernelStartWeightsPtr;

                        var vcInputs = Vector.LoadAligned(currentInputsPtr);

                        // Check if whole kernel is zero => skip sparse activations.
                        if (!Vector.EqualsAll(Vector<float>.Zero, vcInputs))
                        {
                            Vector<float> addents = Vector.LoadAligned(currentBufferPtr);

                            for (int ri = 0; ri < KERNEL_SIZE_IN_FLOATS; ri++)
                            {
                                Vector<float> vecWeights = Vector.LoadAligned(currentWeightsPtr);

                                float x = *currentInputsPtr; // Load the input from inputs
                                addents = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights, addents);

                                // move the weigths point to next line.
                                currentWeightsPtr += KERNEL_SIZE_IN_FLOATS;

                                currentInputsPtr += 1; // Increas inputs pointer for next line
                            }

                            Vector.Store(addents, currentBufferPtr); // Store the result back in the buffer.
                        }
                        else
                        {
                            Vector.Store(Vector<float>.Zero, currentBufferPtr);

                            currentWeightsPtr += KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                            currentInputsPtr += KERNEL_SIZE_IN_FLOATS;
                        }

                        // Move the buffer we work on.
                        currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                    }
                }
            }

            // Signal that this thread has finished the multiplication and wait for the others to finish as well.
            if (Interlocked.Increment(ref *item.arrivedThreadCount) != partitions)
            { // only if we are not the last thread to arive, we wait

                while (Volatile.Read(ref *item.arrivedThreadCount) != partitions)
                {
                    // Here we are on the very hot path of the code, so we want to avoid yielding the thread if possible.
                    Thread.SpinWait(1);
                }
            }

            int cols = hKernelsPerPartition;
            // Distribute the remaining coloumns to the first few partitions.
            if (partitionId < hKernelsRemaining)
                cols++;

            int offset = partitionId * hKernelsPerPartition * KERNEL_SIZE_IN_FLOATS +
                         (partitionId < hKernelsRemaining ? partitionId : hKernelsRemaining) * KERNEL_SIZE_IN_FLOATS;

            currentBufferPtr = ptrRawBuffer + offset * batches;

            // Offset between intermediary results in the buffer
            int bufferOffset = batches * hKernels * KERNEL_SIZE_IN_FLOATS;

            float* startBiasPtr = ptrBias + offset;

            //If there are no coloumns to process we can skip this step and let the other threads do their work.
            if (cols != 0)
            {
                float* startActivationPtr = ptrActivations + offset * batches;

                #region SumLoop
                // The loop is unrolled and this handles p=0 (there must be at least one, else the partiton would not exist)
                // START of the loop p = 0
                {
                    float* startBufferPtr = currentBufferPtr;
                    float* currentActivationPtr = startActivationPtr;
                    float* currentBiasPtr = startBiasPtr;

                    if (partitions - 1 == 0 && applyReLU)
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent = Vector.LoadAligned(currentBiasPtr);
                                Vector<float> buf = Vector.LoadAligned(currentBufferPtr);

                                var res = Vector.Add(buf, addent);
                                res = Vector.Max(res, Vector<float>.Zero); // ReLU
                                res.StoreAligned(currentActivationPtr);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    else
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent = Vector.LoadAligned(currentBiasPtr);
                                Vector<float> buf = Vector.LoadAligned(currentBufferPtr);

                                var res = Vector.Add(buf, addent);
                                res.StoreAligned(currentActivationPtr);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.
                }

                // For the result of each partition add the intermediar results togehter.
                for (int p = 1; p < partitions - 1; p++)
                {
                    float* startBufferPtr = currentBufferPtr;
                    float* currentActivationPtr = startActivationPtr;
                    float* currentBiasPtr = startBiasPtr;

                    for (int c = 0; c < cols; c++)
                    {
                        for (int b = 0; b < batches; b++)
                        {

                            Vector<float> addent = Vector.LoadAligned(currentActivationPtr);
                            Vector<float> buf = Vector.LoadAligned(currentBufferPtr);
                            var res = Vector.Add(buf, addent);

                            res.StoreAligned(currentActivationPtr);

                            currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                            currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                        currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                    }
                    currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.
                }

                // The end of the loop unrolled p = partitions -1
                if (partitions >= 2)
                {
                    float* startBufferPtr = currentBufferPtr;
                    float* currentActivationPtr = startActivationPtr;
                    float* currentBiasPtr = startBiasPtr;

                    // The code is here a near duplicate but that is accaptable as the 
                    // compiler is so better able to compile good IL and from there Assembly and Machine Instructions
                    // as the branching is moved outside of the loop and the body of the loop ramins branchless
                    if (applyReLU)
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {
                                Vector<float> addent = Vector.LoadAligned(currentActivationPtr);
                                Vector<float> buf = Vector.LoadAligned(currentBufferPtr);

                                var res = Vector.Add(buf, addent);
                                res = Vector.Max(res, Vector<float>.Zero); // ReLU
                                res.StoreAligned(currentActivationPtr);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    else
                    {
                        for (int c = 0; c < cols; c++)
                        {
                            for (int b = 0; b < batches; b++)
                            {

                                Vector<float> addent = Vector.LoadAligned(currentActivationPtr);
                                Vector<float> buf = Vector.LoadAligned(currentBufferPtr);
                                var res = Vector.Add(buf, addent);

                                res.StoreAligned(currentActivationPtr);

                                currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                    currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.
                }
                #endregion
            }
        }

        /// <inheritdoc/>
        public Task FusedMultiplyAdd(int batches, int[] weightsShape, Span<float> inputs, Span<float> weights, Span<float> bias, NativeMemoryOwner<float> activations, bool applyReLU = true, CancellationToken ct = default)
        {
            Debug.Assert(inputs.Length % 16 == 0 && weights.Length % 16 == 0 && bias.Length % 16 == 0, "The shape of the tensors must be divisible by 16");
            // Making sure we can actually do the calculation and it doesn't crash on machines that don't support AVX512 but 256 bit register.
            Debug.Assert(KERNEL_SIZE_IN_FLOATS % Vector<float>.Count == 0, "KERNEL_SIZE must be a multiple of Vector<T>.Count.");

            // From the view of the weights tensor.
            int vKernels = weightsShape[0] / KERNEL_SIZE_IN_FLOATS;
            int hKernels = weightsShape[1] / KERNEL_SIZE_IN_FLOATS;

            tasks.Clear();
            if (threadedResults is not null)
                // Return the buffer to the buffer manager. (This happens when called to clean up before the next invocation)
                threadedResults.Dispose();

            unsafe
            {
                // Convert the spans to points so we can work with them for the hardware accelerated caclulations
                ref float rWeights = ref MemoryMarshal.GetReference(weights);
                ref float rInputs = ref MemoryMarshal.GetReference(inputs);
                ref float rBias = ref MemoryMarshal.GetReference(bias);
                ref float rActivations = ref MemoryMarshal.GetReference(activations.Data);

                float* ptrInputs = (float*)Unsafe.AsPointer(ref rInputs);
                float* ptrWeights = (float*)Unsafe.AsPointer(ref rWeights);
                float* ptrBias = (float*)Unsafe.AsPointer(ref rBias);
                float* ptrActivations = (float*)Unsafe.AsPointer(ref rActivations);

                int* arrivedThreadCountPtr = threadedResultsArrivedCount.Pointer;
                *arrivedThreadCountPtr = 0;

                int elementsInVector = Vector<float>.Count;

                SFMAWorkItem* workItems = workItemBuffer.Pointer;

                // Distribute the work evenly 
                int vKernelsPerPartition = vKernels / threadPool.NumberOfThreads;
                int vKernelsRemaining = vKernels % threadPool.NumberOfThreads;

                int partitions = vKernelsPerPartition == 0 ? vKernelsRemaining : threadPool.NumberOfThreads;

                int offsetWeights = 0;
                int offsetInputs = 0;

                threadedResults = bufferManager.GetBuffer(activations.Data.Length * threadPool.NumberOfThreads, sizeof(float));
                var threadedResultsBuffer = threadedResults.Buffer.Slice(0, activations.Data.Length * partitions);

                var rawBuffer = threadedResultsBuffer;

                // Get a pointer to said buffer.
                ref float rRawBuffer = ref MemoryMarshal.GetReference(rawBuffer);
                float* ptrRawBuffer = (float*)Unsafe.AsPointer(ref rRawBuffer);
                float* ptrBuffer = ptrRawBuffer;

                // For each partition produce the intermediary result of x*W
                for (int i = 0; i < partitions; i += 1)
                {
                    var vHeightInPartition = vKernelsPerPartition;
                    if (vKernelsRemaining > 0)
                    {
                        vHeightInPartition++;
                        vKernelsRemaining--;
                    }

                    // Calculate the pointers this work-item needs.
                    float* currentWeightsPtr = ptrWeights + offsetWeights;
                    float* currentInputsPtr = ptrInputs + offsetInputs;
                    float* currentBufferPtr = ptrBuffer;

                    int partitionId = i;

                    var workItem = workItems + i;
                    *workItem = new SFMAWorkItem
                    {
                        currentInputsPtr = currentInputsPtr,
                        currentBufferPtr = currentBufferPtr,
                        currentWeightsPtr = currentWeightsPtr,
                        ptrRawBuffer = ptrRawBuffer,
                        ptrBias = ptrBias,
                        ptrActivations = ptrActivations,
                        hKernels = hKernels,
                        vHeightInPartition = vHeightInPartition,
                        batches = batches,
                        partitionId = partitionId,
                        partitions = partitions,
                        applyReLU = applyReLU,
                        arrivedThreadCount = arrivedThreadCountPtr,
                    };

                    Task? sfmaTask;
                    if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                    {
                        sfmaTask = threadPool.Schedule(&FusedMultiplyAddReLUWorkNative, workItem);
                    }
                    else
                    {
                        sfmaTask = threadPool.Schedule(&FusedMultiplyAddReLUWorkSequential, workItem);

                    }

                    if (sfmaTask is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks.Add(sfmaTask);

                    offsetWeights += vHeightInPartition * hKernels * KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                    offsetInputs += vHeightInPartition * KERNEL_SIZE_IN_FLOATS * batches;

                    ptrBuffer += activations.Data.Length;
                }

            }

            return Task.WhenAll(tasks);
        }

        public void Dispose()
        {
        }
    }
}
