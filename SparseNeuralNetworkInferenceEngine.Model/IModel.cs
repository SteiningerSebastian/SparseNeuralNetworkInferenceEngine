using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public interface IModel
    {
        /// <summary>
        /// Invokes the model to return the result.
        /// </summary>
        /// <param name="tensor">The tensor for the input of the model.</param>
        /// <returns>Returns the result as a tensor.</returns>
        public Task<Tensor<float>> InvokeAsync(Tensor<float> tensor);

        /// <summary>
        /// Compiles the model for execution.
        /// </summary>
        public void Compile();

        /// <summary>
        /// Loads the model layer by layer by loading weights, bias, weights, ... bias.
        /// </summary>
        /// <param name="parameters">An enumerable with all parameters of the model.</param>
        public void Load(IEnumerable<float> parameters);

        /// <summary>
        /// Loads the model from a given binary file.
        /// </summary>
        /// <param name="path">The path to the binary file.</param>
        public Task LoadAsync(string path);

        /// <summary>
        /// Stores the model parameters in binary format.
        /// </summary>
        /// <returns>The binary values of the Model</returns>
        public float[] Store();

        /// <summary>
        /// Stores the model at the given location.
        /// </summary>
        /// <param name="path">The path to store the model at.</param>
        /// <returns>Returns the task that finishes once the model is stored.</returns>
        public Task StoreAsync(string path);
    }
}
