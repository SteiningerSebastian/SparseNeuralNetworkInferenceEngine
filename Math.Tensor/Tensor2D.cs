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
        /// <param name="activations">The result, a tensor stored in the BatchValueTensorLayout.</param>
        /// <param name="applyReLU">If the ReLU activation function should be applied</param>
        public async Task SparseFusedMultiplyAdd(Tensor2D<float> weights, Tensor1D<float> bias, Tensor2D<float> activations, bool applyReLU)
        {
            if (typeof(T) != typeof(float))
                throw new NotImplementedException("Sparse Fused Multiply Add is only supported with single precision floating point numbers.");

            // Making sure the layout of the Tensor matches the supported layout for the tensors
            Debug.Assert(weights.LayoutMapper.GetType() == typeof(WeightsTensorMemoryLayout), "Expected WeightsTensorMemoryMapper for weight tensor.");
            Debug.Assert(bias.LayoutMapper.GetType() == typeof(RowMajorTensorMemoryLayout), "Expected RowMajorTensorMemoryLayout for bais tensor.");
            Debug.Assert(activations.LayoutMapper.GetType() == typeof(BatchValueTensorMemoryLayout), "Expected BatchValueTensorMemoryMapper for result tensor.");

            // Making sure their shape matches.
            Debug.Assert(weights.Shape[0] == Shape[1], $"Unable to multiply tensors of shape ({string.Join(',', Shape)}) and ({string.Join(',', Shape)}).");
            Debug.Assert(Shape[0] == activations.Shape[0], $"Unable to store result in tensor of shape ({string.Join(',', activations.Shape)})");

            Debug.Assert(weights.Shape[0] % 16 == 0 && weights.Shape[1] % 16 == 0, "Shape of weights must be divisible by 16.");

            Debug.Assert(accelerator is ISparseFusedMultiplyAddReLU, "This Operation is only supported with an hardware accelerator as its slow otherwise.");


            if (accelerator is ISparseFusedMultiplyAddReLU acc)
            {
                Span<float> inputs = MemoryMarshal.Cast<T, float>(data.Data);

                await acc.FusedMultiplyAdd(Shape[0], weights.Shape, inputs, weights.data.Data, bias.GetValues(), activations.data, applyReLU);

#if DEBUG
                // For debugging it is very usefull to make sure the result is correct for each hardware
                // accelerated operation to make sure not to propagate errors. 
                // Comment this code in and out as needed, it is very expensive to run and should never be used in production.
                #region Debugging Assert
                //var assertActivations = activations.DeepCopy();

                //for (int batch = 0; batch < Shape[0]; batch++)
                //{
                //    // Copy bias to activations
                //    for(int act = 0; act < assertActivations.Shape[1]; act++)
                //    {
                //        assertActivations[batch, act] = bias[act];
                //    }

                //    Tensor2D<float> inputsT = (Tensor2D<float>)(object)this;

                //    // Perform the matrix multiplication
                //    for (int act = 0; act < assertActivations.Shape[1]; act++)
                //    {
                //        for(int i = 0; i < weights.Shape[0]; i++)
                //        {
                //            assertActivations[batch, act] += inputsT[batch, i] * weights[i, act];
                //        }
                //    }

                //    // Aply ReLU if needed
                //    if (applyReLU)
                //    {
                //        for (int act = 0; act < assertActivations.Shape[1]; act++)
                //        {
                //            assertActivations[batch, act] = MathF.Max(0, assertActivations[batch, act]);
                //        }
                //    }
                //}

                //for(int batch = 0; batch < Shape[0]; batch++)
                //{
                //    for( int act = 0; act < assertActivations.Shape[1]; act++)
                //    {
                //        Debug.Assert(MathF.Abs(assertActivations[batch, act] - activations[batch, act]) < 0.01, "Hardware accelerated result does not match expected activations.");
                //    }
                //}
                #endregion
#endif

                return;
            }

            throw new NotImplementedException();
        }

        public override Tensor<T> DynamicCast(int[] shape)
        {
            Debug.Assert(shape.Length == 2, $"Operation not supported for tensor of shape ({string.Join(',', shape)}).");
            Debug.Assert(shape[1] <= this.Shape[1], "May only dynamically cast tensor into smaller tensor");

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
                for (int batch = 0; batch < this.Shape[0]; batch++)
                {
                    var enumerable = EnumerateValuesInBatch(batch);
                    enumerable = function(enumerable);

                    int i = 0;
                    foreach (var v in enumerable)
                    {
                        this[batch, i] = v;
                        i++;
                    }

                }
            }
            else
            {
                await base.ApplyFunction(function);
            }
        }
    }
}
