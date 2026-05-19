using SparseNeuralNetworkInferenceEngine.Math.Tensor;

namespace SparseNeuralNetworkInferenceEngine.Math
{
    public interface IActivationFunction
    {
        /// <summary>
        /// The activation function that is applied.
        /// </summary>
        public void Invoke(Tensor<float> values);
    }
}
