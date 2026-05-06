using SparseNeuralNetworkInferenceEngine.General;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SparseNeuralNetworkInferenceEngine.Math.Tensor
{
    public class Tensor2D<T> : Tensor1D<T> where T : unmanaged
    {
        public Tensor2D() : base() { }

        /// <summary>
        /// Creates a new one dimensional tensor.
        /// </summary>
        /// <param name="d1">The size of the tensor along the first dimension.</param>
        /// <param name="d2">The size of the tensor along the second dimension.</param>
        /// <param name="initialize">True if the tensor should be initialized.</param>
        /// <param name="alligned">True if the tensor should be alligned to a cache line.</param>
        /// <param name="pageAlligned">True if the tensor should be alligned to a page boundry.</param>
        /// <param name="values">The values to initialize the thensor to.</param>
        public Tensor2D(int d1, int d2, ITensorMemoryLayout mapper, bool initialize = false, bool alligned = true, bool pageAlligned = false, IEnumerable<T>? values = null, IHardwareAccelerator? accelerator = null) :
            base(d1 * d2, initialize, alligned, pageAlligned, null)
        {
            this.accelerator = accelerator;
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
        public async Task SparseFusedMultiplyAdd(Tensor2D<float> weights, Tensor1D<float> bias, Tensor2D<float> result)
        {
            if (typeof(T) != typeof(float))
                throw new NotImplementedException("Sparse Fused Multiply Add is only supported with single precision floating point numbers.");

            // Making sure the layout of the Tensor matches the supported layout for the tensors
            Debug.Assert(weights.LayoutMapper.GetType() == typeof(WeightsTensorMemoryLayout), "Expected WeightsTensorMemoryMapper for weight tensor.");
            Debug.Assert(bias.LayoutMapper.GetType() == typeof(RowMajorTensorMemoryLayout), "Expected RowMajorTensorMemoryLayout for bais tensor.");
            Debug.Assert(result.LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout), "Expected BatchValueTensorMemoryMapper for result tensor.");

            // Making sure their shape matches.
            Debug.Assert(weights.Shape[0] == Shape[1], $"Unable to multiply tensors of shape ({string.Join(',', Shape)}) and ({string.Join(',', Shape)}).");
            Debug.Assert(Shape[0] == result.Shape[0], $"Unable to store result in tensor of shape ({string.Join(',', result.Shape)})");

            Debug.Assert(weights.Shape[0] % 16 == 0 && weights.Shape[1] % 16 == 0, "Shape of weights must be divisible by 16.");

            Debug.Assert(accelerator is ISparseFusedMultiplyAddReLU, "This Operation is only supported with an hardware accelerator as its slow otherwise.");

            if (accelerator is ISparseFusedMultiplyAddReLU acc)
            {
                Span<float> inputs = MemoryMarshal.Cast<T, float>(data.Data);

                await acc.FusedMultiplyAdd(shape[0], weights.shape, inputs, weights.data.Data, bias.GetValues(), result.data);
                return;
            }

            throw new NotImplementedException(); // Not implemented without using hardware acceleration because very slow...
        }

        public override Tensor<T> DynamicCast(int[] shape)
        {
            Debug.Assert(shape.Length == 2, $"Operation not supported for tensor of shape ({string.Join(',', shape)}).");
            Debug.Assert(shape[1] < this.Shape[1], "May only dynamically cast tensor into smaller tensor");

            if (LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout))
            {
                var o = (Tensor2D<T>)this.Clone();
                o.shape = [Shape[0], shape[1]];
                return o;
            }

            throw new NotImplementedException();
        }

        /// <summary>
        /// Enumerates all elements in the given batch.
        /// </summary>
        /// <param name="batch">The batch to enumerate.</param>
        /// <returns></returns>
        protected IEnumerable<T> EnumerateValuesInBatch(int batch)
        {
            Debug.Assert(this.LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout));
            for (int i = 0; i < this.Shape[1]; i++)
            {
                yield return this[batch, i];
            }
        }

        /// <summary>
        /// Applies the given function to the tensor. If the tensor is in 
        /// BatchValue-Layout, the function is multithreaded and applied across each
        /// single batch.
        /// </summary>
        /// <param name="function">The function to apply</param>
        /// <returns>A task that complets if all operations complete.</returns>
        public async override Task ApplyFunction(Func<IEnumerable<T>, IEnumerable<T>> function)
        {
            if (LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout))
            {
                for (int batch = 0; batch < this.Shape.Length; batch++)
                {
                    var enumerable = EnumerateValuesInBatch(batch);
                    var enumerator = function(enumerable).GetEnumerator();

                    for (int i = 0; i < Shape[1]; i++)
                    {
                        this[batch, i] = enumerator.Current;
                        enumerator.MoveNext();
                    }
                }
            }

            await base.ApplyFunction(function);
        }
    }
}
