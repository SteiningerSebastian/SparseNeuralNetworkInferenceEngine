using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Math.Tensor
{
    public class IdentityTensorMemoryMapper : ITensorMemoryMapper
    {

        public IdentityTensorMemoryMapper(){}

        public object Clone()
        {
            return new IdentityTensorMemoryMapper();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int MapToMemory(int[] index)
        {
            Debug.Assert(index.Length == 1, "Identity wrapper expects an one dimensional input");
            return index[0];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int[] MapToTensor(int offset)
        {
            return [offset];
        }
    }
}
