using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Math.Tensor
{
    public class Tensor1D<T> : Tensor<T> where T: INumber<T>
    {
        /// <summary>
        /// Creates a new one dimensional tensor.
        /// </summary>
        /// <param name="length">The size of the one dimensional Tensor to create.</param>
        /// <param name="initialize">True if the tensor should be initialized.</param>
        /// <param name="alligned">True if the tensor should be alligned to a cache line.</param>
        /// <param name="pageAlligned">True if the tensor should be alligned to a page boundry.</param>
        /// <param name="values">The values to initialize the thensor to.</param>
        public Tensor1D(int length, bool initialize = false, bool alligned = true, bool pageAlligned = false, IEnumerable<T>? values = null) : base()
        {
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

        public Tensor1D() : base() { } 
    }
}
