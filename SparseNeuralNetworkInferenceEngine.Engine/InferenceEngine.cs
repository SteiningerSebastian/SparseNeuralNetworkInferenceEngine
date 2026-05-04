using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using SparseNeuralNetworkInferenceEngine.General;

namespace SparseNeuralNetworkInferenceEngine.Engine
{
    public class InferenceEngine : IInferenceEngine 
    {
        protected IHardwareAccelerator? accelerator;

        public InferenceEngine(IHardwareAccelerator? accelerator = null)
        {
            this.accelerator = accelerator;
        }


        /// <summary>
        /// Allocates a new tensor.
        /// </summary>
        /// <typeparam name="T">Either float or double depending of the desire precision.</typeparam>
        /// <param name="shape">The shape of the tensor to allocate.</param>
        /// <param name="layout">The memory layout of the tensor.</param>
        /// <param name="initialize">Whether to initialize the underlying memory.</param>
        /// <param name="alignment">Whether to align the memory to cache lines.</param>
        /// <param name="pageAlignment">Wheter to align thememory to page boundries. (Warning: only use for large tensors)</param>
        /// <param name="values">If set the enumerable to load into the tensor. </param>
        /// <returns>The allocated tensor is returned.</returns>
        /// <exception cref="ArgumentException">Is thrown if a tensor of the given shape can't be created.</exception>
        public T AllocateTensor<T,K>(int[] shape, ITensorMemoryLayout layout, bool initialize, bool alignment, bool pageAlignment, IEnumerable<K>? values) where T : ITensor<K> where K: unmanaged
        {
            if (shape.Length == 0)
            {
                throw new ArgumentException($"Unable to create a tensor of shape ({string.Join(',', shape)}).");
            }

            T tensor;
            // Factory for tensors.
            if (typeof(T) == typeof(Tensor1D<K>))
            {
                tensor = (T)(object)new Tensor1D<K>(shape[0], initialize, alignment, pageAlignment, values, accelerator);
            }
            else if (typeof(T) == typeof(Tensor2D<K>))
            {
                tensor = (T)(object)new Tensor2D<K>(shape[0], shape[1], layout, initialize, alignment, pageAlignment, values, accelerator);
            }
            else
            {
                throw new ArgumentException($"Unable to create a tensor of shape ({string.Join(',', shape)}).");
            }

            return tensor;
        }

        /// <summary>
        /// Allocates a new tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateTensor<T,K>(ITensorMemoryLayout layout, params int[] shape) where T : ITensor<K> where K: unmanaged =>
            AllocateTensor<T,K>(shape, layout, true, false, false, null);

        /// <summary>
        /// Allocates a new aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateAlignedTensor<T,K>(params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), true, true, false, null);

        /// <summary>
        /// Allocates a new aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateAlignedTensor<T,K>(ITensorMemoryLayout layout, params int[] shape) where T : ITensor<K> where K: unmanaged =>
            AllocateTensor<T,K>(shape, layout, true, true, false, null);

        /// <summary>
        /// Allocates a new tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateTensor<T,K>(params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), true, false, false, null);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedTensor<T,K>(ITensorMemoryLayout layout, params int[] shape) where T : ITensor<K> where K: unmanaged =>
            AllocateTensor<T,K>(shape, layout, false, false, false, null);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedTensor<T,K>(params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), false, false, false, null);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedTensor<T,K>(IEnumerable<K> values, params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), false, false, false, values);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedTensor<T,K>(ITensorMemoryLayout layout, IEnumerable<K> values, params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, layout, false, false, false, values);


        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedAlignedTensor<T,K>(ITensorMemoryLayout layout, params int[] shape) where T : ITensor<K> where K: unmanaged =>
            AllocateTensor<T,K>(shape, layout, false, true, false, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedAlignedTensor<T,K>(params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), false, true, false, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedAlignedTensor<T,K>(IEnumerable<K> values, params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), false, true, false, values);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedAlignedTensor<T,K>(ITensorMemoryLayout layout, IEnumerable<K> values, params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, layout, false, true, false, values);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedPageAlignedTensor<T,K>(ITensorMemoryLayout layout, params int[] shape) where T : ITensor<K> where K: unmanaged =>
            AllocateTensor<T,K>(shape, layout, false, true, true, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedPageAlignedTensor<T,K>(params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), false, true, true, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedPageAlignedTensor<T,K>(IEnumerable<K> values, params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, new RowMajorTensorMemoryLayout(shape), false, true, true, values);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public T AllocateUninitializedPageAlignedTensor<T,K>(ITensorMemoryLayout layout, IEnumerable<K> values, params int[] shape) where T : ITensor<K> where K: unmanaged =>
           AllocateTensor<T,K>(shape, layout, false, true, true, values);
    }
}
