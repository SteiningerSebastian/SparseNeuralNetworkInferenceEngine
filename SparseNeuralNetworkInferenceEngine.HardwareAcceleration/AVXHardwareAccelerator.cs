using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.HardwareAcceleration
{
    public class AVXHardwareAccelerator : IHardwareAccelerator, IAddAligned, ISparseFusedMultiplyAddReLU
    {
        protected IThreadPool threadPool;

        public AVXHardwareAccelerator(IThreadPool threadPool)
        {
            this.threadPool = threadPool;

            if (Vector<float>.Count <= 4)
            {
                throw new HardwareAccelerationException("This hardware accelerator requieres support for 256bit or 512bit Avx / SIMD!");
            }
        }

        /// <inheritdoc/>
        public Task AddAsync<T>(Span<T> addend1, Span<T> addend2) where T : INumber<T>
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
                addend1[i] += addend2[i];
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
                    var t = threadPool.Schedule((i) =>
                    {

                        for (int cio = co; cio < end; cio++)
                        {
                            // If 8 floats / 512bit are supported we can do this in a single instruction. If the machine does not support 512 bit but 256bit
                            // do two instructions unroled.
                            if (Vector<float>.Count == KERNEL_SIZE_IN_FLOATS)
                            {
                                Vector<float> vecBias = Vector.LoadAligned(ptrBias + cio * elementsInVector);
                                for (int b = 0; b < batches; b++)
                                {
                                    Vector.Store(vecBias, ptrActivations + batches * elementsInVector + cio * (batches * elementsInVector));
                                }
                            }
                            else
                            {
                                Vector<float> vecBias1 = Vector.LoadAligned(ptrBias + cio * elementsInVector);
                                Vector<float> vecBias2 = Vector.LoadAligned(ptrBias + cio * elementsInVector + Vector<float>.Count);

                                for (int b = 0; b < batches; b++)
                                {
                                    Vector.Store(vecBias1, ptrActivations + batches * elementsInVector + cio * (batches * elementsInVector));
                                    Vector.Store(vecBias2, ptrActivations + batches * elementsInVector + cio * (batches * elementsInVector) + Vector<float>.Count);

                                }
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
        public Task FusedMultiplyAddReLU<T>(int batches, int[] weightsShape, Span<T> inputs, Span<T> weights, Span<T> bias, Span<T> activations, CancellationToken ct = default)
        {
            Debug.Assert(typeof(T) == typeof(float), "Only single precision floating point numbers are supported");
            Debug.Assert(inputs.Length % 16 == 0 && weights.Length % 16 == 0 && bias.Length % 16 == 0, "The shape of the tensors must be divisible by 16");
            // Making sure we can actually do the calculation and it doesn't crash on machines that don't support AVX512 but 256 bit register.
            Debug.Assert(KERNEL_SIZE_IN_FLOATS % Vector<T>.Count == 0, "KERNEL_SIZE must be a multiple of Vector<T>.Count.");

            // From the view of the weights tensor.
            int vKernels = weightsShape[0] / KERNEL_SIZE_IN_FLOATS;
            int hKernels = weightsShape[1] / KERNEL_SIZE_IN_FLOATS;

            Task task;

            unsafe
            {
                ref T rWeights = ref MemoryMarshal.GetReference(weights);
                ref T rInputs = ref MemoryMarshal.GetReference(inputs);
                ref T rBias = ref MemoryMarshal.GetReference(bias);
                ref T rActivations = ref MemoryMarshal.GetReference(activations);

                float* ptrInputs = (float*)Unsafe.AsPointer(ref rInputs);
                float* ptrWeights = (float*)Unsafe.AsPointer(ref rWeights);
                float* ptrBias = (float*)Unsafe.AsPointer(ref rBias);
                float* ptrActivations = (float*)Unsafe.AsPointer(ref rActivations);

                int elementsInVector = Vector<T>.Count;

                var tasks = new List<Task>();

                var copyBiasTask = CopyBiasToActivations(batches, hKernels, elementsInVector, ptrBias, ptrActivations, ct);
                tasks.Add(copyBiasTask);

                int partitions = Math.Min(threadPool.NumberOfThreads, vKernels);
                int vKernelsPerPartition = (int)Math.Ceiling(vKernels / (float)partitions);

                int offset = 0;
                // Add the values
                for (int i = 0; i < partitions; i += 1)
                {
                    int startOffset = offset;
                    int endOffset = offset + vKernelsPerPartition * hKernels * KERNEL_SIZE_IN_FLOATS * KERNEL_SIZE_IN_FLOATS;

                    var th = threadPool.Schedule((_) =>
                    {
                       //TODO: Here the calculation happens for each intermediary result.
                    });

                    if (th is null)
                        throw new HardwareAccelerationException("Failed to shedule operations. (Operation Overflow)");

                    tasks.Add(th);

                    offset = endOffset;
                }

                task = Task.WhenAll(tasks).ContinueWith(async (t) =>
                {
                    // TODO: sum up the final result of the calculation
                });
                
            }

            return task;
        }
    }
}
