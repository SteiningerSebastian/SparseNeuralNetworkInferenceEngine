using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public sealed class NativeMemoryBuffer<T> : IDisposable where T: unmanaged
    {
        private NativeMemoryOwner<T>? memoryOwner;

        private NativeMemoryBufferManager<T> manager;

        private int size = 0;

        /// <summary>
        /// The buffer that can be used to store data.
        /// </summary>
        public Span<T> Buffer => memoryOwner!.Data.Slice(0, size);

        /// <summary>
        /// Creates a new Memory Buffer with a memoyr owner and manager.
        /// </summary>
        public NativeMemoryBuffer(NativeMemoryOwner<T> memoryOwner, NativeMemoryBufferManager<T> manager, int size)
        {
            this.memoryOwner = memoryOwner;
            this.manager = manager;
            this.size = size;
        }

        public void Clear()
        {
            memoryOwner?.Data.Clear();
        }

        public void Dispose()
        {
            manager.FreeBuffer(this);
            memoryOwner = null; // get rid of references
            manager = null!;
        }

        /// <summary>
        /// Releases the internal NativeMemoryOwner. Warning accessing the Buffer will fail after this operation.
        /// </summary>
        /// <returns>Returns the release nativeMemoryOwner.</returns>
        public NativeMemoryOwner<T> ReleaseMemory()
        {
            if (memoryOwner is null)
                throw new NullReferenceException("Unable to release NativeMemoryOwner. NativeMemoryOwner was already released.");

            var mo = memoryOwner;
            memoryOwner = null;
            return mo;
        }
    }
}
