using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Numerics;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public class NativeMemoryBufferManager<T> where T : unmanaged
    {
        /// <summary>
        /// During the calculation of the hardware accelerated fused multiply add we need certain buffers, to not reallocate use a stack
        /// take buffers from the stack if exist and if no longer needed put them back.
        /// Sizes always a multiple of two / bags to ensure proper reuse of memory.
        /// </summary>
        protected ConcurrentDictionary<nuint, ConcurrentStack<NativeMemoryOwner<T>>> buffers;

        protected int count = 0;

        /// <summary>
        /// The current number of bufferse managed.
        /// </summary>
        public int Count => count;

        protected int maxSize = -1;

        /// <summary>
        /// Creates a new manager for native memory buffers (large buffers on the hot path)
        /// </summary>
        /// <param name="maxSize">The maximum number of buffers to manage. -1 for infinite</param>
        public NativeMemoryBufferManager(int maxSize)
        {
            this.maxSize = maxSize;
            buffers = new ConcurrentDictionary<nuint, ConcurrentStack<NativeMemoryOwner<T>>>();
        }

        public NativeMemoryBuffer<T> GetBuffer(int size, int typeSize)
        {
            // Round up to the next multiple of 2
            var bucket = BitOperations.RoundUpToPowerOf2((nuint)size);

            var stack = buffers.GetOrAdd(bucket, (i) => { return new ConcurrentStack<NativeMemoryOwner<T>>(); });

            if(stack.TryPop(out var owner))
            {
                return new NativeMemoryBuffer<T>(owner, this, size);
            }

            if (maxSize != -1 && Count >= maxSize)
                throw new InvalidOperationException("Reached max-size, can not create new buffer. All buffers are beeing used.");

            Interlocked.Increment(ref count);

            owner = new NativeMemoryOwner<T>(bucket, (nuint)typeSize, 64);
            return new NativeMemoryBuffer<T>(owner, this, size);
        }
        


        /// <summary>
        /// This method is called automatically by the NativeMemoryBuffer on dispose to return the NativeMemoryOwner.
        /// </summary>
        /// <param name="buffer">The buffer that is returned</param>
        public void FreeBuffer(NativeMemoryBuffer<T> buffer)
        {
            var owner = buffer.ReleaseMemory();
            int size = owner.Data.Length;
            var bucket = BitOperations.RoundUpToPowerOf2((nuint)size);

            if(buffers.TryGetValue(bucket, out var stack))
            {
                stack.Push(owner);
            }
            else
            {
                throw new InvalidOperationException("Unable to find bucket to return NativeMemoryOwner to");
            }
        }
    }
}
