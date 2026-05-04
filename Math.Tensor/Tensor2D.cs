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
        public Tensor2D(int d1, int d2, ITensorMemoryLayout mapper, bool initialize = false, bool alligned = true, bool pageAlligned = false, IEnumerable<T>? values = null) :
            base(d1 * d2, initialize, alligned, pageAlligned, null)
        {
            shape = [d1, d2];
            this.LayoutMapper = mapper;

            if (values != null)
            {
                PopulateWithEnumerable(values);
            }
        }

        /// <summary>
        /// Calculates the result of x * W + b and stores it in the result Tensor using SIMD instructions.
        /// </summary>
        /// <param name="weights">The weights tensor stored in WeightTensorMemoryLayout.</param>
        /// <param name="bias">The bias tensor stored in the BatchValueTensorLayout.</param>
        /// <param name="result">The result, a tensor stored in the BatchValueTensorLayout.</param>
        public void FusedMultiplyAddCPU(Tensor2D<T> weights, Tensor1D<T> bias, Tensor2D<T> result) {
            // Making sure the layout of the Tensor matches the supported layout for the tensors
            Debug.Assert(weights.LayoutMapper.GetType() ==  typeof(WeightsTensorMemoryMapper), "Expected WeightsTensorMemoryMapper for weight tensor.");
            Debug.Assert(bias.LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout), "Expected BatchValueTensorMemoryMapper for bais tensor.");
            Debug.Assert(result.LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout), "Expected BatchValueTensorMemoryMapper for result tensor.");

            // Making sure their shape matches.
            Debug.Assert(weights.Shape[0] == Shape[1], $"Unable to multiply tensors of shape ({string.Join(',', Shape)}) and ({string.Join(',', Shape)}).");
            Debug.Assert(Shape[0] == result.Shape[0], $"Unable to store result in tensor of shape ({string.Join(',', result.Shape)})");
            Debug.Assert(bias.Shape[0] == Shape[0], $"Unable to add tensor of shape ({string.Join(',', bias.Shape)}) to tensor of shape ({string.Join(',', result.Shape)}).");

            //TODO: MOVE TO ENGINE ....
            
        }
    }
}
