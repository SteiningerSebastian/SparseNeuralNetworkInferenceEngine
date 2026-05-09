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
    public class AVXHardwareAccelerator : IHardwareAccelerator, IAddAligned, ISparseFusedMultiplyAddReLU
    {
        protected IThreadPool threadPool;
        protected NativeMemoryBufferManager<float> bufferManager;

        public AVXHardwareAccelerator(IThreadPool threadPool)
        {
            this.threadPool = threadPool;

            bufferManager = new NativeMemoryBufferManager<float>(-1);

            if (Vector<float>.Count <= 4)
            {
                throw new HardwareAccelerationException("This hardware accelerator requieres support for 256bit or 512bit Avx / SIMD!");
            }
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

        /// <inheritdoc/>
        public Task AddAsync(Span<float> addend1, Span<float> addend2)
        {
            // Can only add elements of same length.
            Debug.Assert(addend1.Length == addend2.Length, "Only spans of same size are supported.");
            Debug.Assert(addend1.Length % KERNEL_SIZE_IN_FLOATS == 0, "The vectors muust have a length of a multiple of KERNEL_SIZE_IN_FLOATS.");

            int nVectors = addend1.Length / KERNEL_SIZE_IN_FLOATS;
            Task t;

            unsafe
            {
                ref float add1 = ref MemoryMarshal.GetReference(addend1);
                ref float add2 = ref MemoryMarshal.GetReference(addend2);

                float* ptrAdd1 = (float*)Unsafe.AsPointer(ref add1);
                float* ptrAdd2 = (float*)Unsafe.AsPointer(ref add2);

                int vectorsPerPartition = nVectors / threadPool.NumberOfThreads;
                int vectorsRemaining = nVectors % threadPool.NumberOfThreads;

                int requiredWorkers = vectorsPerPartition == 0 ? vectorsRemaining : threadPool.NumberOfThreads;

                Task[] tasks = new Task[requiredWorkers];

                int offset = 0;
                // Add the values with the required workers, each has at least 1 item.
                for (int i = 0; i < requiredWorkers; i += 1)
                {
                    int workItems = vectorsPerPartition;

                    // Distribute the remaining work to the first
                    // few workers. 
                    if (vectorsRemaining > 0)
                    {
                        workItems++;
                        vectorsRemaining--;
                    }

                    float* currentAdd1Ptr = ptrAdd1 + offset;
                    float* currentAdd2Ptr = ptrAdd2 + offset;

                    var th = threadPool.Schedule((_, ct) =>
                    {
                        for (int k = 0; k < workItems; k++)
                        {
                            if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                            {
                                var v1 = Vector.LoadAligned(currentAdd1Ptr);
                                var v2 = Vector.LoadAligned(currentAdd2Ptr);
                                Vector.Add(v1, v2).StoreAligned(currentAdd1Ptr);
                            }
                            else
                            {
                                var v1 = Vector.LoadAligned(currentAdd1Ptr);
                                var v2 = Vector.LoadAligned(currentAdd2Ptr);
                                Vector.Add(v1, v2).StoreAligned(currentAdd1Ptr);

                                var v3 = Vector.LoadAligned(currentAdd1Ptr + Vector<float>.Count);
                                var v4 = Vector.LoadAligned(currentAdd2Ptr + Vector<float>.Count);
                                Vector.Add(v3, v4).StoreAligned(currentAdd1Ptr + Vector<float>.Count);
                            }

                            currentAdd1Ptr += KERNEL_SIZE_IN_FLOATS;
                            currentAdd2Ptr += KERNEL_SIZE_IN_FLOATS;
                        }
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks[i] = th;

                    offset += workItems * KERNEL_SIZE_IN_FLOATS;
                }

                t = Task.WhenAll(tasks);

            }

            return t;
        }


        /// <summary>
        /// While this function is basically the same as the one above, to avoid branching inside such high performance code
        /// we created this function again with the only difference that Max(0, element) is applied before storing in addend1
        /// </summary>
        protected Task AddReLUAsync(Span<float> addend1, Span<float> addend2)
        {
            // Can only add elements of same length.
            Debug.Assert(addend1.Length == addend2.Length, "Only spans of same size are supported.");
            Debug.Assert(addend1.Length % KERNEL_SIZE_IN_FLOATS == 0, "The vectors muust have a length of a multiple of KERNEL_SIZE_IN_FLOATS.");

            int nVectors = addend1.Length / KERNEL_SIZE_IN_FLOATS;
            Task t;

            unsafe
            {
                ref float add1 = ref MemoryMarshal.GetReference(addend1);
                ref float add2 = ref MemoryMarshal.GetReference(addend2);

                float* ptrAdd1 = (float*)Unsafe.AsPointer(ref add1);
                float* ptrAdd2 = (float*)Unsafe.AsPointer(ref add2);

                int vectorsPerPartition = nVectors / threadPool.NumberOfThreads;
                int vectorsRemaining = nVectors % threadPool.NumberOfThreads;

                int requiredWorkers = vectorsPerPartition == 0 ? vectorsRemaining : threadPool.NumberOfThreads;

                Task[] tasks = new Task[requiredWorkers];

                int offset = 0;
                // Add the values with the required workers, each has at least 1 item.
                for (int i = 0; i < requiredWorkers; i += 1)
                {
                    int workItems = vectorsPerPartition;

                    // Distribute the remaining work to the first
                    // few workers. 
                    if (vectorsRemaining > 0)
                    {
                        workItems++;
                        vectorsRemaining--;
                    }

                    float* currentAdd1Ptr = ptrAdd1 + offset;
                    float* currentAdd2Ptr = ptrAdd2 + offset;

                    var th = threadPool.Schedule((_, ct) =>
                    {
                        var zeros = Vector.Create<float>(0);
                        for (int k = 0; k < workItems; k++)
                        {
                            if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                            {
                                var v1 = Vector.LoadAligned(currentAdd1Ptr);
                                var v2 = Vector.LoadAligned(currentAdd2Ptr);
                                var res = Vector.Add(v1, v2);
                                res = Vector.Max(res, zeros);
                                res.StoreAligned(currentAdd1Ptr);
                            }
                            else
                            {
                                var v1 = Vector.LoadAligned(currentAdd1Ptr);
                                var v2 = Vector.LoadAligned(currentAdd2Ptr);
                                var res1 = Vector.Add(v1, v2);
                                res1 = Vector.Max(res1, zeros);
                                res1.StoreAligned(currentAdd1Ptr);

                                var v3 = Vector.LoadAligned(currentAdd1Ptr + Vector<float>.Count);
                                var v4 = Vector.LoadAligned(currentAdd2Ptr + Vector<float>.Count);
                                var res2 = Vector.Add(v3, v4);
                                res2 = Vector.Max(res2, zeros);
                                res2.StoreAligned(currentAdd1Ptr + Vector<float>.Count);
                            }

                            currentAdd1Ptr += KERNEL_SIZE_IN_FLOATS;
                            currentAdd2Ptr += KERNEL_SIZE_IN_FLOATS;
                        }
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks[i] = th;

                    offset += workItems * KERNEL_SIZE_IN_FLOATS;
                }

                t = Task.WhenAll(tasks);

            }

            return t;
        }


        const int KERNEL_SIZE_IN_FLOATS = (Settings.KERNEL_SIZE / sizeof(float));


        /// <summary>
        /// Copies the batch to the activations.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="batches"></param>
        /// <param name="hKernels"></param>
        /// <param name="elementsInVector"></param>
        /// <param name="tasks"></param>
        /// <param name="ptrBias"></param>
        /// <param name="ptrActivations"></param>
        /// <param name="ct"></param>
        /// <returns>A task that completes when all events complete.</returns>
        /// <exception cref="HardwareAccelerationException"></exception>

        protected unsafe Task CopyBiasToActivations(int batches, int hKernels, float* ptrBias, float* ptrActivations, CancellationToken ct = default)
        {
            List<Task> tasks = new();

            // Distribute the work evenly   
            int hKernelsPerThread = hKernels / threadPool.NumberOfThreads;
            int hKernelsRemaing = hKernels % threadPool.NumberOfThreads;

            int requiredWorkers = hKernelsPerThread == 0 ? hKernelsRemaing : threadPool.NumberOfThreads;

            int offsetBias = 0;
            int offsetActivations = 0;
            // Copy the bais to the activations
            for (int c = 0; c < requiredWorkers; c++)
            {
                int hKernelsInTask = hKernelsPerThread;
                if (hKernelsRemaing > 0)
                {
                    hKernelsInTask++;
                    hKernelsRemaing--;
                }

                float* currentBiasPtr = ptrBias + offsetBias;
                float* currentActivationPtr = ptrActivations + offsetActivations;
                var t = threadPool.Schedule((i, ct) =>
                {

                    for (int k = 0; k < hKernelsInTask; k++)
                    {
                        // If 8 floats / 512bit are supported we can do this in a single instruction. If the machine does not support 512 bit but 256bit
                        // do two instructions unroled.
                        if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                        {
                            Vector<float> vecBias = Vector.LoadAligned(currentBiasPtr);
                            for (int b = 0; b < batches; b++)
                            {
                                Vector.Store(vecBias, currentActivationPtr);
                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                        else
                        {
                            Vector<float> vecBias1 = Vector.LoadAligned(currentBiasPtr);
                            Vector<float> vecBias2 = Vector.LoadAligned(currentBiasPtr + Vector<float>.Count);

                            for (int b = 0; b < batches; b++)
                            {
                                Vector.Store(vecBias1, currentActivationPtr);
                                Vector.Store(vecBias2, currentActivationPtr + Vector<float>.Count);

                                currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                            }
                            currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                        }
                    }
                });

                if (t is null)
                {
                    throw new HardwareAccelerationException("Unable to schedule calculation.");
                }

                tasks.Add(t);

                offsetBias += KERNEL_SIZE_IN_FLOATS * hKernelsInTask;
                offsetActivations += KERNEL_SIZE_IN_FLOATS * hKernelsInTask * batches;
            }


            return Task.WhenAll(tasks);
        }


        /// <inheritdoc/>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public Task FusedMultiplyAdd(int batches, int[] weightsShape, Span<float> inputs, Span<float> weights, Span<float> bias, NativeMemoryOwner<float> activations, bool applyReLU = true, CancellationToken ct = default)
        {
            Debug.Assert(inputs.Length % 16 == 0 && weights.Length % 16 == 0 && bias.Length % 16 == 0, "The shape of the tensors must be divisible by 16");
            // Making sure we can actually do the calculation and it doesn't crash on machines that don't support AVX512 but 256 bit register.
            Debug.Assert(KERNEL_SIZE_IN_FLOATS % Vector<float>.Count == 0, "KERNEL_SIZE must be a multiple of Vector<T>.Count.");

            // From the view of the weights tensor.
            int vKernels = weightsShape[0] / KERNEL_SIZE_IN_FLOATS;
            int hKernels = weightsShape[1] / KERNEL_SIZE_IN_FLOATS;

            var tasks = new List<Task>(threadPool.NumberOfThreads);
            Task task;
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

                int elementsInVector = Vector<float>.Count;


                // Distribute the work evenly 
                int vKernelsPerPartition = vKernels / threadPool.NumberOfThreads;
                int vKernelsRemaining = vKernels % threadPool.NumberOfThreads;

                int partitions = vKernelsPerPartition == 0 ? vKernelsRemaining : threadPool.NumberOfThreads;

                int offsetWeights = 0;
                int offsetInputs = 0;

                var threadedResults = bufferManager.GetBuffer(activations.Data.Length * threadPool.NumberOfThreads, sizeof(float));
                var threadedResultsBuffer = threadedResults.Buffer.Slice(0, activations.Data.Length * partitions);

                var addBarrier = new Barrier(partitions);

                // For each partition produce the intermediary result of x*W
                for (int i = 0; i < partitions; i += 1)
                {
                    var vHeightInPartition = vKernelsPerPartition;
                    if (vKernelsRemaining > 0)
                    {
                        vHeightInPartition++;
                        vKernelsRemaining--;
                    }

                    var rawBuffer = threadedResultsBuffer;

                    // Get a pointer to said buffer.
                    ref float rRawBuffer = ref MemoryMarshal.GetReference(rawBuffer);
                    float* ptrRawBuffer = (float*)Unsafe.AsPointer(ref rRawBuffer);
                    float* ptrBuffer = ptrRawBuffer + i * activations.Data.Length;

                    // Calculate the pointers this work-item needs.
                    float* currentWeightsPtr = ptrWeights + offsetWeights;
                    float* currentInputsPtr = ptrInputs + offsetInputs;
                    float* currentBufferPtr = ptrBuffer;

                    int partitionId = i;
                    int hKernelsPerPartition = hKernels / threadPool.NumberOfThreads;
                    int hKernelsRemaining = hKernels % threadPool.NumberOfThreads;

                    var th = threadPool.Schedule((_, ct) =>
                    {
                        float* coloumnStartInputsPtr = currentInputsPtr;

                        for (int c = 0; c < hKernels; c++)
                        {
                            float* coloumnStartBufferPtr = currentBufferPtr;
                            currentInputsPtr = coloumnStartInputsPtr;

                            // The amount of rows we need to compute before moving on the the next coloumn
                            for (int r = 0; r < vHeightInPartition; r += 1)
                            {
                                // these rows are in the same part of the buffer
                                currentBufferPtr = coloumnStartBufferPtr;

                                float* kernelStartWeightsPtr = currentWeightsPtr;

                                // This kernel for every batch to keep weights in cahce
                                for (int b = 0; b < batches; b++)
                                {
                                    // For the batch start at the beginning of the weights (weights and inputs)
                                    currentWeightsPtr = kernelStartWeightsPtr;

                                    if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                                    {
                                        var vcInputs = Vector.LoadAligned(currentInputsPtr);

                                        // Check if whole kernel is zero => skip sparse activations.
                                        if (!Vector.EqualsAll(Vector<float>.Zero, vcInputs))
                                        {

                                            Vector<float> addents;
                                            // For the first row we need to initialize the buffer with zero, for the subsequent rows we must load it.
                                            // TODO This could be unrolled but the JIT should be able to handle it. -> Check performance and only unroll if it is a bottleneck.
                                            if (r == 0)
                                            {
                                                addents = Vector<float>.Zero;
                                            }
                                            else
                                            {
                                                addents = Vector.LoadAligned(currentBufferPtr);
                                            }

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
                                            if (r == 0)
                                            {
                                                Vector.Store(Vector<float>.Zero, currentBufferPtr);
                                            }

                                            currentWeightsPtr += KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                                            currentInputsPtr += KERNEL_SIZE_IN_FLOATS;
                                        }
                                    }
                                    else // Again for machines that only support AVX2 / a 256 bit register do the work in sequence.
                                    // This should be reordered by the JIT to work best on the current plattform
                                    {
                                        var vcInputs1 = Vector.LoadAligned(currentInputsPtr);
                                        var vcInputs2 = Vector.LoadAligned(currentInputsPtr + Vector<float>.Count);

                                        // Check if whole kernel is zero => skip sparse activations.
                                        if (!(Vector.EqualsAll(Vector<float>.Zero, vcInputs1) && Vector.EqualsAll(Vector<float>.Zero, vcInputs2)))
                                        {
                                            Vector<float> addents1;
                                            Vector<float> addents2;

                                            // For the first row we need to initialize the buffer with zero, for the subsequent rows we must load it.
                                            // TODO This could be unrolled but the JIT should be able to handle it. -> Check performance and only unroll if it is a bottleneck.
                                            if (r == 0)
                                            {
                                                addents1 = Vector<float>.Zero;
                                                addents2 = Vector<float>.Zero;
                                            }
                                            else
                                            {
                                                addents1 = Vector.LoadAligned(currentBufferPtr);
                                                addents2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                            }

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
                                            if(r == 0)
                                            {
                                                Vector.Store(Vector<float>.Zero, currentBufferPtr);
                                                Vector.Store(Vector<float>.Zero, currentBufferPtr + Vector<float>.Count);
                                            }

                                            currentWeightsPtr += KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                                            currentInputsPtr += KERNEL_SIZE_IN_FLOATS;
                                        }
                                    }

                                    // Move the buffer we work on.
                                    currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                }
                            }
                        }

                        // Check if all threads have cleared the barrier and the buffer is ready to be added together.
                        addBarrier.SignalAndWait(ct);

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

                        // If there are no coloumns to process we can skip this step and let the other threads do their work.
                        if (cols != 0)
                        {
                            float* startActivationPtr = ptrActivations + offset * batches;
                            // For the result of each partition add the intermediar results togehter.
                            for (int p = 0; p < partitions; p++)
                            {
                                float* startBufferPtr = currentBufferPtr;
                                float* currentActivationPtr = startActivationPtr;
                                float* currentBiasPtr = startBiasPtr;

                                for (int c = 0; c < cols; c++)
                                {
                                    for (int b = 0; b < batches; b++)
                                    {
                                        if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                                        {
                                            Vector<float> addent;
                                            Vector<float> buf;

                                            // TODO This could be unrolled but the JIT should be able to handle it. -> Check performance and only unroll if it is a bottleneck.
                                            if (p == 0)
                                            {
                                                addent = Vector.LoadAligned(currentBiasPtr);
                                                buf = Vector.LoadAligned(currentBufferPtr);
                                            }
                                            else
                                            {
                                                addent = Vector.LoadAligned(currentActivationPtr);
                                                buf = Vector.LoadAligned(currentBufferPtr);
                                            }

                                            var res = Vector.Add(buf, addent);
                                            // TODO This could be unrolled but the JIT should be able to handle it. -> Check performance and only unroll if it is a bottleneck.
                                            if (applyReLU && p == partitions - 1)
                                            {
                                                res = Vector.Max(res, Vector<float>.Zero);
                                            }

                                            res.StoreAligned(currentActivationPtr);
                                        }
                                        else
                                        {
                                            Vector<float> addent1;
                                            Vector<float> addent2;
                                            Vector<float> buf1;
                                            Vector<float> buf2;

                                            // TODO This could be unrolled but the JIT should be able to handle it. -> Check performance and only unroll if it is a bottleneck.
                                            if (p == 0)
                                            {
                                                addent1 = Vector.LoadAligned(currentBiasPtr);
                                                addent2 = Vector.LoadAligned(currentBiasPtr + Vector<float>.Count);
                                                buf1 = Vector.LoadAligned(currentBufferPtr);
                                                buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                            }
                                            else
                                            {
                                                addent1 = Vector.LoadAligned(currentActivationPtr);
                                                addent2 = Vector.LoadAligned(currentActivationPtr + Vector<float>.Count);
                                                buf1 = Vector.LoadAligned(currentBufferPtr);
                                                buf2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);
                                            }

                                            var res1 = Vector.Add(buf1, addent1);
                                            var res2 = Vector.Add(buf2, addent2);
                                            
                                            // TODO This could be unrolled but the JIT should be able to handle it. -> Check performance and only unroll if it is a bottleneck.
                                            if (applyReLU && p == partitions - 1)
                                            {
                                                res1 = Vector.Max(res1, Vector<float>.Zero);
                                                res2 = Vector.Max(res2, Vector<float>.Zero);
                                            }

                                            res1.StoreAligned(currentActivationPtr);
                                            res2.StoreAligned(currentActivationPtr + Vector<float>.Count);
                                        }
                                        currentBufferPtr += KERNEL_SIZE_IN_FLOATS;
                                        currentActivationPtr += KERNEL_SIZE_IN_FLOATS;
                                    }
                                    currentBiasPtr += KERNEL_SIZE_IN_FLOATS;
                                }
                                currentBufferPtr = startBufferPtr + bufferOffset; // Move to the next partition in the buffer.
                            }
                        }
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks.Add(th);

                    offsetWeights += vHeightInPartition * hKernels * KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;
                    offsetInputs += vHeightInPartition * KERNEL_SIZE_IN_FLOATS * batches;
                }

                // Making sure we dispose the buffer after all threads have completed their work.
                task = Task.WhenAll(tasks).ContinueWith(_ => threadedResults.Dispose());
            }

            return task;
            ;
        }
    }
}
