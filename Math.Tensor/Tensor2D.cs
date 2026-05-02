using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Text;

namespace Math.Tensor
{
    public class Tensor2D<T> : Tensor1D<T> where T : INumber<T>
    {
        /// <summary>
        /// Creates a new one dimensional tensor.
        /// </summary>
        /// <param name="d1">The size of the tensor along the first dimension.</param>
        /// <param name="d2">The size of the tensor along the second dimension.</param>
        /// <param name="initialize">True if the tensor should be initialized.</param>
        /// <param name="alligned">True if the tensor should be alligned to a cache line.</param>
        /// <param name="pageAlligned">True if the tensor should be alligned to a page boundry.</param>
        /// <param name="values">The values to initialize the thensor to.</param>
        public Tensor2D(int d1, int d2, ITensorMemoryMapper mapper, bool initialize = false, bool alligned = true, bool pageAlligned = false, IEnumerable<T>? values = null) :
            base(d1 * d2, initialize, alligned, pageAlligned, null)
        {
            shape = [d1, d2];
            this.mapper = mapper;

            if (values != null)
            {
                PopulateWithEnumerable(values);
            }
        }
    }
}
