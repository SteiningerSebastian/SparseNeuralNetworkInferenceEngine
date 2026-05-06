using SparseNeuralNetworkInferenceEngine.General;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace SparseNeuralNetworkInferenceEngine.Math.Tensor
{
    public class Tensor1D<T> : Tensor<T> where T: unmanaged
    {
        public Tensor1D():base(){}

        /// <summary>
        /// Creates a new one dimensional tensor.
        /// </summary>
        /// <param name="length">The size of the one dimensional Tensor to create.</param>
        /// <param name="initialize">True if the tensor should be initialized.</param>
        /// <param name="alligned">True if the tensor should be alligned to a cache line.</param>
        /// <param name="pageAlligned">True if the tensor should be alligned to a page boundry.</param>
        /// <param name="values">The values to initialize the thensor to.</param>
        public Tensor1D(int length, bool initialize = false, bool alligned = true, bool pageAlligned = false, IEnumerable<T>? values = null, IHardwareAccelerator? accelerator = null) : base()
        {
            this.accelerator = accelerator;
            shape = [length];
            LayoutMapper = new RowMajorTensorMemoryLayout(shape);

            // Can't allign memory to page boundry but not cache line.
            Debug.Assert(!(pageAlligned && !alligned));

            this.Length = length;

            nuint typeSize = (nuint)(typeof(T) == typeof(float) ? 4 : 8);

            nuint alignment = (nuint)IntPtr.Size; // Standard alignment for the system. (32 vs 64 bit machines)
            if (alligned)
            {
                alignment = Settings.CACHE_LINE_SIZE;
            }

            if (pageAlligned)
            {
                alignment = (nuint)Environment.SystemPageSize;
            }

            data = new NativeMemoryOwner<T>((nuint)length, typeSize, alignment);

            if (initialize && values is null)
            {
                data.Data.Clear(); // Zero the whole array.
            }

            // If values are given, copy them to the Data.
            values?.ToArray().CopyTo(data.Data);
        }

        /// <summary>
        /// Define what happens when another element is added. (+= only as result is stored in this tensor)
        /// </summary>
        /// <param name="operand">The operator to add.</param>
        public async Task AddAsync(Tensor1D<T> operand)
        {
            if(this.accelerator is not null)
            {
                // Acceleration is only supported for floats
                if(this.accelerator is IAddAligned acc && typeof(T) == typeof(float))
                {
                    Span<float> a = MemoryMarshal.Cast<T, float>(data.Data);
                    Span<float> b = MemoryMarshal.Cast<T, float>(operand.data.Data);
                    await acc.AddAsync(a, b);
                    return;
                }
            }

            // Simple non accelerated operation.
            for (int i = 0; i < this.Length; i++)
            {
                if (typeof(T) == typeof(float))
                {
                    var a1 = MemoryMarshal.Cast<T, float>(this.data.Data);
                    var a2 = MemoryMarshal.Cast<T, float>(operand.data.Data);
                    a1[i] += a2[i];

                }
                else
                {
                    var a1 = MemoryMarshal.Cast<T, double>(this.data.Data);
                    var a2 = MemoryMarshal.Cast<T, double>(operand.data.Data);
                    a1[i] += a2[i];
                }
            }
        }

        /// <inheritdoc/>
        public override Tensor<T> DynamicCast(int[] shape)
        {
            Debug.Assert(shape.Length == 1, $"Operation not supported for tensor of shape ({string.Join(',', shape)}).");
            Debug.Assert(shape[0] < this.Shape[0], "May only dynamically cast tensor into smaller tensor");

            if(LayoutMapper.GetType() == typeof(RowMajorTensorMemoryLayout))
            {
                var o = (Tensor1D<T>)this.Clone();
                // We just decrease the size of the tensor but not the memory.
                o.Length = shape[0];
                return o;
            }

            throw new NotImplementedException();
        }

        public Span<T> GetValues() => data.Data;
    }
}
