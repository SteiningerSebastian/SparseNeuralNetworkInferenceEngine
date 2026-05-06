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

        public ModelSequential(List<ILayer> layers, IInferenceEngine engine)
        {
            Debug.Assert(layers[0] is InputLayer, "The first layer must be of type InputLayer");
            Debug.Assert(layers[layers.Count - 1] is OutputLayer, "The last layer must be of type OutputLayer");

            this.layers = layers;
            this.engine = engine;
        }

        public void Compile()
        {
            int[] shape = [];
            foreach (ILayer layer in layers)
            {
                shape = layer.Compile(shape, engine);
            }
        }

        public async Task<Tensor<float>> InvokeAsync(Tensor<float> tensor)
        {
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
        }
    }
}
