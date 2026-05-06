using SparseNeuralNetworkInferenceEngine.Math.Tensor;

namespace SparseNeuralNetworkInferenceEngine.Math
{
    public interface IActivationFunction
    {
        /// <summary>
        /// The activation function that is applied.
        /// </summary>
        /// <param name="values">An enumerable over all values.</param>
        /// <returns>The enumerable over all values.</returns>
        public IEnumerable<float> Invoke(IEnumerable<float> values);
    }
}
