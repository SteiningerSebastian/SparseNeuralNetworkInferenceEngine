using Math.Tensor;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace SparseNEuronalNetworkInferenceEngine.Engine
{
    public class InferenceEngine : IInferenceEngine
    {
        public IThreadPool ThreadPool { get; }

        public InferenceEngine(int queueCapacity = 1024, ThreadPriority threadPriority = ThreadPriority.Normal, int? threads = null, CancellationToken ct = default)
        {
            // if the number of threads was not specified, use number of logical cpu cores.
            if (threads is null)
            {
                threads = Environment.ProcessorCount;
            }

            ThreadPool = new ThreadPool((int)threads, queueCapacity, threadPriority, ct);
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
        public Tensor<T> AllocateTensor<T>(int[] shape, ITensorMemoryLayout layout, bool initialize, bool alignment, bool pageAlignment, IEnumerable<T>? values) where T : INumber<T>
        {
            if (shape.Length == 0)
            {
                throw new ArgumentException($"Unable to create a tensor of shape ({string.Join(',', shape)}).");
            }
            Tensor<T> tensor;

            // Factory for tensors.
            if (shape.Length == 1)
            {
                tensor = new Tensor1D<T>(shape[0], initialize, alignment, pageAlignment, values);
            }
            else if (shape.Length == 2)
            {
                tensor = new Tensor2D<T>(shape[0], shape[1], layout, initialize, alignment, pageAlignment, values);
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
        public Tensor<T> AllocateTensor<T>(ITensorMemoryLayout layout, params int[] shape) where T : INumber<T> =>
            AllocateTensor<T>(shape, layout, true, false, false, null);

        /// <summary>
        /// Allocates a new aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateAlignedTensor<T>(params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), true, true, false, null);

        /// <summary>
        /// Allocates a new aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateAlignedTensor<T>(ITensorMemoryLayout layout, params int[] shape) where T : INumber<T> =>
            AllocateTensor<T>(shape, layout, true, true, false, null);

        /// <summary>
        /// Allocates a new tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateTensor<T>(params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), true, false, false, null);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedTensor<T>(ITensorMemoryLayout layout, params int[] shape) where T : INumber<T> =>
            AllocateTensor<T>(shape, layout, false, false, false, null);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedTensor<T>(params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), false, false, false, null);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedTensor<T>(IEnumerable<T> values, params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), false, false, false, values);

        /// <summary>
        /// Allocates a new uninitialized tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedTensor<T>(ITensorMemoryLayout layout, IEnumerable<T> values, params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, layout, false, false, false, values);


        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedAlignedTensor<T>(ITensorMemoryLayout layout, params int[] shape) where T : INumber<T> =>
            AllocateTensor<T>(shape, layout, false, true, false, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedAlignedTensor<T>(params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), false, true, false, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedAlignedTensor<T>(IEnumerable<T> values, params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), false, true, false, values);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedAlignedTensor<T>(ITensorMemoryLayout layout, IEnumerable<T> values, params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, layout, false, true, false, values);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double depending on the desired precision.</typeparam>
        /// <param name="layout">The layout of the tensor in memory.</param>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedPageAlignedTensor<T>(ITensorMemoryLayout layout, params int[] shape) where T : INumber<T> =>
            AllocateTensor<T>(shape, layout, false, true, true, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedPageAlignedTensor<T>(params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), false, true, true, null);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedPageAlignedTensor<T>(IEnumerable<T> values, params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, new RowMajorTensorMemoryLayout(shape), false, true, true, values);

        /// <summary>
        /// Allocates a new uninitialized aligned tensor with the given layout and shape.
        /// </summary>
        /// <typeparam name="T">Float or Double dpending on the desired precision.</typeparam>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">An enumerable with the values of the tensor in row-major layout.</param>
        /// <returns>The allocated tensor is returned.</returns>
        public Tensor<T> AllocateUninitializedPageAlignedTensor<T>(ITensorMemoryLayout layout, IEnumerable<T> values, params int[] shape) where T : INumber<T> =>
           AllocateTensor<T>(shape, layout, false, true, true, values);
    }
}
