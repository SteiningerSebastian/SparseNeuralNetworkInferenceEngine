using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public interface ILayer
    {
        /// <summary>
        /// Compiles the layer.
        /// </summary>
        /// <param name="inputShape">The shape of the tensor the model receives.</param>
        /// <param name="engine">The inference engine to use.</param>
        /// <returns>The shape of the tensor it produces.</returns>
        public int[] Compile(int[] inputShape, IInferenceEngine engine);

        /// <summary>
        /// Invoke the calculation for the layer.
        /// </summary>
        /// <typeparam name="T">The type to in</typeparam>
        /// <param name="engine">The engine to use for calculating the result.</param>
        /// <param name="tensor">The result.</param>
        public Task<Tensor<float>> InvokeAsync(Tensor<float> tensor, IInferenceEngine engine);

        /// <summary>
        /// Loads the Layer from a given enumerator over modell parameters.
        /// </summary>
        /// <param name="parameters">The parameters of the model.</param>
        public void Load(IEnumerator<float> parameters);

        /// <summary>
        /// Stores the models parameters in the provided store.
        /// </summary>
        /// <param name="store">The store to store the model parameters to.</param>
        /// <returns>The number of floats stored.</returns>
        public int Store(Span<float> store);

        /// <summary>
        /// Returns the number of parameters of a layer.
        /// </summary>
        /// <returns></returns>
        public int NumerOfParameters();

    }
}
