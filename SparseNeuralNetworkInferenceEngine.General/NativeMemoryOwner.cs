using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    // Inspired by: https://learn.microsoft.com/en-us/dotnet/api/system.buffers.memorymanager-1?view=net-10.0

    /// <summary>
    /// Holds the native memory for the tensors.
    /// </summary>
    /// <typeparam name="T">The type to store.</typeparam>
    public unsafe class NativeMemoryOwner<T> : MemoryManager<T>, ICloneable
    {
        private readonly void* pointer;
        private readonly nuint length;
        private readonly nuint typeSize;
        private readonly nuint alignment;
        private bool disposed = false;

        /// <summary>
        /// Get a Span to access the data.
        /// </summary>
        public Span<T> Data => GetSpan();

        public T* Pointer
        {
            get
            {
                if (disposed)
                {
                    throw new ObjectDisposedException(GetType().Name);
                }
                return (T*)pointer;
            }
        }


        /// <inheritdoc/>
        public NativeMemoryOwner(nuint length, nuint typeSize, nuint alignment)
        {
            this.length = length;
            this.typeSize = typeSize;
            this.alignment = alignment;
            this.pointer = NativeMemory.AlignedAlloc(length * typeSize, alignment);
        }

        public NativeMemoryOwner(void* pointer, nuint length, nuint typeSize, nuint alignment, ref bool disposed)
        {
            this.pointer = pointer;
            this.length = length;
            this.typeSize = typeSize;
            this.alignment = alignment;
            this.disposed = disposed;
        }



        /// <inheritdoc/>
        public override Span<T> GetSpan()
        {
            if (disposed) 
                throw new ObjectDisposedException(GetType().FullName);

            return new Span<T>(pointer, (int)length);
        }


        /// <inheritdoc/>
        public override MemoryHandle Pin(int elementIndex = 0) => new MemoryHandle((T*)pointer +  elementIndex);

        /// <inheritdoc/>
        public override void Unpin(){}

        /// <inheritdoc/>
        protected override void Dispose(bool disposing){
            if (!disposed) {
                NativeMemory.Free(pointer);
                disposed = true; // Making sure memory can only be freed once.
            }
        }

        public NativeMemoryOwner<T> DeepCopy()
        {
            NativeMemoryOwner<T> owner;
            unsafe
            {
                owner = new NativeMemoryOwner<T>(length, typeSize, alignment);
            }

            // Copy everything over.
            var oData = owner.Data;
            var tData = this.Data;

            tData.CopyTo(oData); // Copy values over.

            return owner;
        }

        public object Clone()
        {
            return new NativeMemoryOwner<T>(pointer, length, typeSize, alignment, ref disposed);
        }
    }
}
