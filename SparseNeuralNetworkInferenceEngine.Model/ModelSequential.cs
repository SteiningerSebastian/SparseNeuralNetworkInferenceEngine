using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public class ModelSequential : IModel
    {
        protected IList<ILayer> layers;
        protected IInferenceEngine engine;
        protected bool compiled = false;

        public ModelSequential(List<ILayer> layers, IInferenceEngine engine)
        {
            Debug.Assert(layers[0] is InputLayer, "The first layer must be of type InputLayer");
            Debug.Assert(layers[layers.Count - 1] is OutputLayer, "The last layer must be of type OutputLayer");

            this.layers = layers;
            this.engine = engine;
        }

        public void Compile()
        {
            compiled = true;
            int[] shape = [];
            foreach (ILayer layer in layers)
            {
                shape = layer.Compile(shape, engine);
            }
        }

        public async Task<Tensor<float>> InvokeAsync(Tensor<float> tensor)
        {
            Debug.Assert(compiled, "Model needs to be compiled before it can execute.");
            // Call to each layer to do its part and calculate the result.
            foreach (ILayer layer in layers)
            {
                tensor = await layer.InvokeAsync(tensor, engine);
            }

            return tensor;
        }

        public void Load(IEnumerable<float> parameters)
        {
            var enumerator = parameters.GetEnumerator();
            foreach (ILayer layer in layers)
            {
                layer.Load(enumerator);
            }
            // TODO comment in
            //if (enumerator.MoveNext())
            //    throw new InvalidOperationException("Unable to load model. Enumerable contains more values than needed. (Check modell shape)");
        }

        /// <inheritdoc/>
        public async Task LoadAsync(string path)
        {
            var par = await BinaryLoader.ReadFileToFloatEnumerableAsync(path);

            Load(par);
        }

        /// <inheritdoc/>
        public float[] Store()
        {
            Debug.Assert(compiled, "Model needs to be compiled before it can be stored.");
            // Count parameters
            int count = 0;
            foreach (ILayer layer in layers)
            {
                count += layer.NumerOfParameters();
            }


            float[] parameters = new float[count];

            int offset = 0;
            // Call to each layer to do its part and calculate the result.
            foreach (ILayer layer in layers)
            {
                offset += layer.Store(parameters.AsSpan().Slice(offset));
            }

            return parameters;
        }

        /// <inheritdoc/>
        public async Task StoreAsync(string path)
        {
            float[] parameters = Store();
            byte[] bytes = new byte[parameters.Length * sizeof(float)];

            for (int i = 0; i < parameters.Length; i++)
            {
                byte[] b = BitConverter.GetBytes(parameters[i]);
                b.CopyTo(bytes, i * sizeof(float));
            }

            await File.WriteAllBytesAsync(path, bytes);
        }
    }
}
