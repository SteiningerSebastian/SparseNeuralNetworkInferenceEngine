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

        /// <inheritdoc/>
        public Task AddAsync<T>(Span<T> addend1, Span<T> addend2) where T : unmanaged
        {
            Debug.Assert(addend1.Length == addend2.Length, "Only spans of same size are supported."); // Can only add elements of same length.
            Debug.Assert(typeof(T) == typeof(float) || typeof(T) == typeof(double), "Only single or double precision floating point numbers are supported");

            int nVectors = addend1.Length / Vector<T>.Count;
            Task t;

            unsafe
            {
                ref T add1 = ref MemoryMarshal.GetReference(addend1);
                ref T add2 = ref MemoryMarshal.GetReference(addend2);

                T* ptrAdd1 = (T*)Unsafe.AsPointer(ref add1);
                T* ptrAdd2 = (T*)Unsafe.AsPointer(ref add2);

                int partitions = Math.Min(threadPool.NumberOfThreads, nVectors);
                int vectorsPerPartition = (int)Math.Ceiling(nVectors / (float)partitions);

                // Avoiding false sharing, making sure the vectorsPerPartiton is a multiple of 2 (for floats) 256 bit register for AVX256
                if ((Vector<float>.Count == 8 || Vector<double>.Count == 4) && (vectorsPerPartition & 0b1) != 0)
                    vectorsPerPartition += 1;

                Task[] tasks = new Task[partitions];

                int offset = 0;
                // Add the values
                for (int i = 0; i < partitions; i += 1)
                {
                    int startOffset = offset;
                    int endOffset = Math.Min((addend1.Length / Vector<T>.Count) * Vector<T>.Count, offset + vectorsPerPartition * Vector<T>.Count);
                    var th = threadPool.Schedule((_) =>
                    {
                        for (int o = startOffset; o < endOffset; o += Vector<T>.Count)
                        {
                            var v1 = Vector.LoadAligned(ptrAdd1 + o);
                            var v2 = Vector.LoadAligned(ptrAdd2 + o);
                            Vector.Add(v1, v2).StoreAligned(ptrAdd1 + o);
                        }
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks[i] = th;

                    offset += Vector<T>.Count * vectorsPerPartition;
                }

                t = Task.WhenAll(tasks);

            }


            // for remaining floats, not able to be proccessed with hardware accelerated instructions.
            for (int i = nVectors * Vector<T>.Count; i < addend1.Length; i++)
            {
                if (typeof(T) == typeof(float))
                {
                    var a1 = MemoryMarshal.Cast<T, float>(addend1);
                    var a2 = MemoryMarshal.Cast<T, float>(addend2);
                    a1[i] += a2[i];

                }
                else
                {
                    var a1 = MemoryMarshal.Cast<T, double>(addend1);
                    var a2 = MemoryMarshal.Cast<T, double>(addend2);
                    a1[i] += a2[i];
                }
            }

            return t;
        }


        /// <summary>
        /// While this function is basically the same as the one above, to avoid branching inside such high performance code
        /// we created this function again with the only difference that Max(0, element) is applied before storing in addend1
        /// </summary>
        protected Task AddReLUAsync<T>(Span<T> addend1, Span<T> addend2) where T : unmanaged
        {
            Debug.Assert(addend1.Length == addend2.Length, "Only spans of same size are supported."); // Can only add elements of same length.
            Debug.Assert(typeof(T) == typeof(float) || typeof(T) == typeof(double), "Only single or double precision floating point numbers are supported");

            int nVectors = addend1.Length / Vector<T>.Count;
            Task t;

            unsafe
            {
                ref T add1 = ref MemoryMarshal.GetReference(addend1);
                ref T add2 = ref MemoryMarshal.GetReference(addend2);

                T* ptrAdd1 = (T*)Unsafe.AsPointer(ref add1);
                T* ptrAdd2 = (T*)Unsafe.AsPointer(ref add2);

                int partitions = Math.Min(threadPool.NumberOfThreads, nVectors);
                int vectorsPerPartition = (int)Math.Ceiling(nVectors / (float)partitions);

                // Avoiding false sharing, making sure the vectorsPerPartiton is a multiple of 2 (for floats) 256 bit register for AVX256
                if ((Vector<float>.Count == 8 || Vector<double>.Count == 4) && (vectorsPerPartition & 0b1) != 0)
                    vectorsPerPartition += 1;

                Task[] tasks = new Task[partitions];

                int offset = 0;
                // Add the values
                for (int i = 0; i < partitions; i += 1)
                {
                    int startOffset = offset;
                    int endOffset = Math.Min((addend1.Length / Vector<T>.Count) * Vector<T>.Count, offset + vectorsPerPartition * Vector<T>.Count);
                    var th = threadPool.Schedule((_) =>
                    {
                        var zeros = Vector.Create<T>(default(T)); // Default for float, double is zero
                        for (int o = startOffset; o < endOffset; o += Vector<T>.Count)
                        {
                            var v1 = Vector.LoadAligned(ptrAdd1 + o);
                            var v2 = Vector.LoadAligned(ptrAdd2 + o);
                            var res = Vector.Add(v1, v2);
                            // Apply relu to result.
                            res = Vector.Max(zeros, res);
                            res.StoreAligned(ptrAdd1 + o);
                        }
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks[i] = th;

                    offset += Vector<T>.Count * vectorsPerPartition;
                }

                t = Task.WhenAll(tasks);

            }


            // for remaining floats, not able to be proccessed with hardware accelerated instructions.
            for (int i = nVectors * Vector<T>.Count; i < addend1.Length; i++)
            {
                if (typeof(T) == typeof(float))
                {
                    var a1 = MemoryMarshal.Cast<T, float>(addend1);
                    var a2 = MemoryMarshal.Cast<T, float>(addend2);
                    a1[i] += a2[i];

                }
                else
                {
                    var a1 = MemoryMarshal.Cast<T, double>(addend1);
                    var a2 = MemoryMarshal.Cast<T, double>(addend2);
                    a1[i] += a2[i];
                }
            }

            return t;
        }

        public object Clone()
        {
            return new AVXHardwareAccelerator(threadPool);
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

        protected unsafe Task CopyBiasToActivations(int batches, int hKernels, int elementsInVector, float* ptrBias, float* ptrActivations, CancellationToken ct = default)
        {
            List<Task> tasks = new();

            if (batches * hKernels * Vector<float>.Count < Settings.SINGEL_THREAD_OPERATION_THRESHOLD)
            {
                // Copy the bais to the activations
                for (int c = 0; c < hKernels; c++)
                {

                    int co = c;
                    Vector<float> vecBias = Vector.LoadAligned(ptrBias + co * elementsInVector);
                    for (int b = 0; b < batches; b++)
                    {
                        Vector.Store(vecBias, ptrActivations + batches * elementsInVector + co * (batches * elementsInVector));
                    }
                }
            }
            else
            {
                int hKernelsPerThread = (int)Math.Ceiling(hKernels / (float)threadPool.NumberOfThreads);

                // Copy the bais to the activations
                for (int c = 0; c < hKernels; c += hKernelsPerThread)
                {
                    int co = c;
                    int end = Math.Min(hKernels, co + hKernelsPerThread);
                    float* currentBiasPtr = ptrBias;
                    float* currentActivationPtr = ptrActivations;
                    var t = threadPool.Schedule((i) =>
                    {

                        for (int cio = co; cio < end; cio++)
                        {
                            // If 8 floats / 512bit are supported we can do this in a single instruction. If the machine does not support 512 bit but 256bit
                            // do two instructions unroled.
                            if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                            {
                                Vector<float> vecBias = Vector.LoadAligned(currentBiasPtr);
                                for (int b = 0; b < batches; b++)
                                {
                                    Vector.Store(vecBias, currentActivationPtr);
                                    currentActivationPtr += elementsInVector;
                                }
                                currentBiasPtr += elementsInVector;
                            }
                            else
                            {
                                Vector<float> vecBias1 = Vector.LoadAligned(currentBiasPtr);
                                Vector<float> vecBias2 = Vector.LoadAligned(currentBiasPtr + Vector<float>.Count);

                                for (int b = 0; b < batches; b++)
                                {
                                    Vector.Store(vecBias1, currentActivationPtr);
                                    Vector.Store(vecBias2, currentActivationPtr + Vector<float>.Count);

                                    currentActivationPtr += elementsInVector;
                                }
                                currentBiasPtr += elementsInVector;
                            }
                        }
                    });

                    if (t is null)
                    {
                        throw new HardwareAccelerationException("Unable to schedule calculation.");
                    }

                    tasks.Add(t);
                }
            }

            return Task.WhenAll(tasks);
        }

        /// <inheritdoc/>
        public Task FusedMultiplyAddReLU(int batches, int[] weightsShape, Span<float> inputs, Span<float> weights, Span<float> bias, NativeMemoryOwner<float> activations, CancellationToken ct = default)
        {
            Debug.Assert(inputs.Length % 16 == 0 && weights.Length % 16 == 0 && bias.Length % 16 == 0, "The shape of the tensors must be divisible by 16");
            // Making sure we can actually do the calculation and it doesn't crash on machines that don't support AVX512 but 256 bit register.
            Debug.Assert(KERNEL_SIZE_IN_FLOATS % Vector<float>.Count == 0, "KERNEL_SIZE must be a multiple of Vector<T>.Count.");

            // From the view of the weights tensor.
            int vKernels = weightsShape[0] / KERNEL_SIZE_IN_FLOATS;
            int hKernels = weightsShape[1] / KERNEL_SIZE_IN_FLOATS;

            Task task;
            List<NativeMemoryBuffer<float>> activationResults = new();
            var tasks = new List<Task>();

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

                // Initializing the result tensor with the bias.
                var copyBiasTask = CopyBiasToActivations(batches, hKernels, elementsInVector, ptrBias, ptrActivations, ct);

                tasks.Add(copyBiasTask);

                int partitions = Math.Min(threadPool.NumberOfThreads, vKernels);
                int vKernelsPerPartition = (int)Math.Ceiling(vKernels / (float)partitions);

                int offset = 0;
                // For each partition produce the intermediary result of x*W
                for (int i = 0; i < partitions; i += 1)
                {
                    int startOffset = offset;
                    int endOffset = offset + vKernelsPerPartition * hKernels * KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;

                    // Buffer for the thread to work with.
                    var buffer = bufferManager.GetBuffer(activations.Data.Length, sizeof(float));

                    activationResults.Add(buffer);

                    // Get a pointer to said buffer.
                    ref float rBuffer = ref MemoryMarshal.GetReference(buffer.Buffer);
                    float* ptrBuffer = (float*)Unsafe.AsPointer(ref rBuffer);

                    // Calculate the pointers this work-item needs.
                    float* currentWeightsPtr = ptrWeights + offset;
                    float* currentInputsPtr = ptrInputs;
                    float* currentBufferPtr = ptrBuffer;

                    var th = threadPool.Schedule((_) =>
                    {
                        buffer.Clear(); // For performance reason the buffer is created uninitialized, so clear it.
                        var zeros = Vector.Create<float>(0);

                        for (int c = 0; c < hKernels; c++)
                        {
                            // The amount of rows we need to compute before moving on the the next coloumn
                            for (int r = 0; r < vKernelsPerPartition; r += 1)
                            {
                                float* kernelStartWeightsPtr = currentWeightsPtr;
                                float* kernelStartInputsPtr = currentInputsPtr;
                                // This kernel for every batch to keep weights in cahce
                                for (int b = 0; b < batches; b++)
                                {
                                    if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                                    {
                                        var vcInputs = Vector.LoadAligned(currentInputsPtr);
                                        // Check if whole kernel is zero => skip sparse activations.
                                        if (Vector.EqualsAll(zeros, vcInputs))
                                            continue;

                                        Vector<float> addents = Vector.LoadAligned(currentBufferPtr);

                                        for (int ri = 0; ri < KERNEL_SIZE_IN_FLOATS; ri++)
                                        {
                                            Vector<float> vecWeights = Vector.LoadAligned(currentWeightsPtr);

                                            float x = *ptrInputs; // Load the input from inputs
                                            addents = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights, addents);

                                            // move the weigths point to next line.
                                            currentWeightsPtr += elementsInVector;

                                            ptrInputs += 1; // Increas inputs pointer for next line
                                        }

                                        Vector.Store(addents, currentBufferPtr); // Store the result back in the buffer.
                                    }
                                    else // Again for machines that only support AVX2 / a 256 bit register do the work in sequence.
                                    // This should be reordered by the JIT to work best on the current plattform
                                    {
                                        var vcInputs1 = Vector.LoadAligned(currentInputsPtr);
                                        var vcInputs2 = Vector.LoadAligned(currentInputsPtr + Vector<float>.Count);
                                        // Check if whole kernel is zero => skip sparse activations.
                                        if (Vector.EqualsAll(zeros, vcInputs1) && Vector.EqualsAll(zeros, vcInputs2))
                                            continue;

                                        Vector<float> addents1 = Vector.LoadAligned(currentBufferPtr);
                                        Vector<float> addents2 = Vector.LoadAligned(currentBufferPtr + Vector<float>.Count);

                                        for (int ri = 0; ri < KERNEL_SIZE_IN_FLOATS; ri++)
                                        {
                                            Vector<float> vecWeights1 = Vector.LoadAligned(currentWeightsPtr);
                                            Vector<float> vecWeights2 = Vector.LoadAligned(currentWeightsPtr + Vector<float>.Count);

                                            float x = *ptrInputs; // Load the input from inputs
                                            addents1 = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights1, addents1);
                                            addents2 = Vector.FusedMultiplyAdd(Vector.Create(x), vecWeights2, addents2);

                                            // move the weigths point to next line.
                                            currentWeightsPtr += elementsInVector;

                                            ptrInputs += 1; // Increas inputs pointer for next line
                                        }

                                        Vector.Store(addents1, currentBufferPtr); // Store the result back in the buffer.
                                        Vector.Store(addents2, currentBufferPtr + Vector<float>.Count); // Store the result back in the buffer.
                                    }
                                    // For the next batch start at the beginning again (weights and inputs)
                                    currentWeightsPtr = kernelStartWeightsPtr;
                                    currentInputsPtr = kernelStartInputsPtr;
                                    
                                    // Move the buffer we work on.
                                    currentBufferPtr += elementsInVector; 
                                }
                            }
                            // Move to the next coloumn kernel wise in buffer.
                            currentBufferPtr += elementsInVector;
                            // Move on the nex inputs.
                            ptrInputs += elementsInVector;
                        }
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks.Add(th);

                    offset = endOffset;
                }
            }

            // Sum the individual results of each thread.
            task = Task.WhenAll(tasks);
            for (int i = 1; i < activationResults.Count; i++)
            {
                task = task.ContinueWith(async (_) =>
                {
                    var t = AddAsync<float>(activationResults[0].Buffer, activationResults[i].Buffer);

                    if (t is null)
                        throw new HardwareAccelerationException("Unable to perform operation, work-queue overflow");

                    await t;

                    activationResults[i].Dispose(); // Don't need the buffer anymore so dispose.
                });
            }

            // Now add them to the result (where the bias is already there)
            task = task.ContinueWith(async (_) =>
            {
                // here we add the buffers to the activations data and apply the relu function to it. max(0, x)
                var t = AddReLUAsync<float>(activations.Data, activationResults[0].Buffer);

                if (t is null)
                    throw new HardwareAccelerationException("Unable to perform operation, work-queue overflow");

                await t;

                activationResults[0].Dispose(); //Don't need this buffer anymore so dispose.
            });

            return task;
        }
    }
}
