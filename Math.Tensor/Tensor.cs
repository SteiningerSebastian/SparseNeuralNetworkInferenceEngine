using SparseNeuralNetworkInferenceEngine.General;
using System.Collections;
using System.Diagnostics;
using System.Numerics;

namespace SparseNeuralNetworkInferenceEngine.Math.Tensor
{
    public abstract class Tensor<T> : ICloneable, IEnumerable<T>, ITensor<T> where T : unmanaged
    {
        /// <summary>
        /// The shape of the Tensor, must be set by the children!!
        /// </summary>
        protected int[] shape = new int[0];

        /// <summary>
        /// Returns the shape of the tensor.
        /// </summary>
        public int[] Shape => shape;

        /// <summary>
        /// The alligned memory to be used to store the tensor.
        /// </summary>
        protected NativeMemoryOwner<T> data;

        /// <summary>
        /// The HardwareAccelerator to use.
        /// </summary>
        protected IHardwareAccelerator? accelerator;

        /// <summary>
        /// Maps logical indexes to memory positions.
        /// </summary>
        public ITensorMemoryLayout LayoutMapper { get; protected internal set; }

        /// <summary>
        /// Allow the indexed access to data.
        /// </summary>
        /// <param name="index">The index to identify elements.</param>
        /// <returns>The element at the given position is returned.</returns>
        public T this[params int[] index]
        {
            get
            {
                return GetValue(index);
            }
            set
            {
                SetValue(value, index);
            }
        }

        /// <summary>
        /// The number of elements in the tensor.
        /// </summary>
        public int Length { get; protected set; }

        /// <summary>
        /// Creates a new Tensor of type T. T must be a single or double precision floating point number.
        /// </summary>
        /// <exception cref="NotSupportedException">Is thrown if T is neither float nor double.</exception>
        public Tensor()
        {
            if (!(typeof(T) == typeof(float) || typeof(T) == typeof(double)))
            {
                throw new NotSupportedException("Only single or double precision floating point numbers are accepted.");
            }
        }

        /// <summary>
        /// Access a value at a given index.
        /// </summary>
        /// <param name="index">The index of the number.</param>
        /// <returns>The value at the index.</returns>
        public virtual T GetValue(params int[] index)
        {
            EnsureIndexShape(index);
            return data.Data[LayoutMapper.MapToMemory(index)];
        }

        /// <summary>
        /// Sets a value at the given index.
        /// </summary>
        /// <param name="value">The value to set.</param>
        /// <param name="index">The index of the element to set.</param>
        public virtual void SetValue(T value, params int[] index)
        {
            EnsureIndexShape(index);
            data.Data[LayoutMapper.MapToMemory(index)] = value;
        }

        /// <summary>
        /// Makes sure the shape of the index matches the shape of the tensor.
        /// </summary>
        /// <param name="index">The index to use to index the tensor.</param>
        /// <exception cref="ArgumentException">Is thrown if the index can't be used to index the tensor.</exception>
        protected virtual void EnsureIndexShape(int[] index)
        {
            if (index.Length != shape.Length)
            {
                throw new ArgumentException($"Can't index an tensor of shape ({string.Join(',', Shape)}) with index [{string.Join(',', index)}].");
            }
        }

        /// <summary>
        /// Check if the given shape matches the current shape.
        /// </summary>
        /// <param name="shape">The shape to validate.</param>
        /// <exception cref="ArgumentException">Is thrown if the shapes don't match.</exception>
        protected virtual void EnsureEqualShape(int[] shape)
        {
            if (shape.Length != this.shape.Length || !shape.SequenceEqual(this.shape))
            {
                throw new ArgumentException($"Operation on tensors of shape ({string.Join(',', this.shape)}) and ({string.Join(',', shape)}) is not supported.");
            }
        }

        /// <summary>
        /// Populate the tensor with the given enumerable.
        /// </summary>
        /// <param name="enumerable">The enumerable to use to populate the tensor.</param>
        public virtual void PopulateWithEnumerable(IEnumerable<T> enumerable)
        {
            var enumerator = enumerable.GetEnumerator();

            // Remember the current positon.
            int[] currentPosition = new int[shape.Length];

            bool next = true;
            var spData = data.Data;
            // Go through every dimension there is.
            while (currentPosition[0] < shape[0])
            {
                next = enumerator.MoveNext();

                Debug.Assert(next, "Enumerable ended without populating whole tensor.");

                // Set the memory to the given value.
                int offset = LayoutMapper.MapToMemory(currentPosition);
                spData[offset] = enumerator.Current;

                // Increas the index in the last dimension.
                currentPosition[currentPosition.Length - 1] += 1;

                // Making sure to continue counting the index to next dimension.
                for (int i = shape.Length - 1; i > 0; i--)
                {
                    if (shape[i] == currentPosition[i])
                    {
                        currentPosition[i] = 0;
                        currentPosition[i - 1] += 1;
                    }
                }
            }
            next = enumerator.MoveNext();
            Debug.Assert(!next, "Unable to fit enumerable into tensor.");
        }

        public virtual object Clone()
        {
            Tensor<T>? obj = (Tensor<T>?)Activator.CreateInstance(this.GetType());
            if (obj is null) throw new InvalidOperationException($"Unable to create object of type {this.GetType()}.");

            obj.data = this.data;
            obj.shape = shape;
            obj.Length = this.Length;
            obj.LayoutMapper = (ITensorMemoryLayout)this.LayoutMapper.Clone();
            obj.accelerator = accelerator;

            return obj;
        }

        /// <summary>
        /// Creates a new tensor that is an exact deep-copy of the original.
        /// </summary>
        /// <returns>The copy is returned.</returns>
        /// <exception cref="InvalidOperationException">Is thrown if no new object of the type could be created.</exception>
        public virtual Tensor<T> DeepCopy()
        {
            Tensor<T>? obj = (Tensor<T>?)Activator.CreateInstance(this.GetType());
            if (obj is null) throw new InvalidOperationException($"Unable to create object of type {this.GetType()}.");

            obj.data = this.data.DeepCopy();
            obj.shape = shape.ToArray();
            obj.Length = this.Length;
            obj.LayoutMapper = (ITensorMemoryLayout)this.LayoutMapper.Clone();
            obj.accelerator = (IHardwareAccelerator?)accelerator?.Clone();

            return obj;
        }

        /// <inheritdoc/>
        public virtual IEnumerator<T> GetEnumerator()
        {
            // Remember the current positon.
            int[] currentPosition = new int[shape.Length];

            // Go through every dimension there is.
            while (currentPosition[0] < shape[0])
            {
                yield return this[currentPosition];

                // Increas the index in the last dimension.
                currentPosition[currentPosition.Length - 1] += 1;

                // Making sure to continue counting the index to next dimension.
                for (int i = shape.Length - 1; i > 0; i--)
                {
                    if (shape[i] == currentPosition[i])
                    {
                        currentPosition[i] = 0;
                        currentPosition[i - 1] += 1;
                    }
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
