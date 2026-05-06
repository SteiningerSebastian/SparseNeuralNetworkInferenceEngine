using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.Math;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.Model
{
    public class ActivationLayer : ILayer
    {
        protected IActivationFunction function;
        public ActivationLayer(IActivationFunction function)
        {
            this.function = function;
        }

        public int[] Compile(int[] inputShape, IInferenceEngine engine)
        {
            return inputShape;
        }

        public async Task<Tensor<float>> InvokeAsync(Tensor<float> tensor, IInferenceEngine engine)
        {
            await tensor.ApplyFunction(function.Invoke);
            return tensor;
        }

        public void Load(IEnumerator<float> parameters)
        {
        }

        public int NumerOfParameters() => 0;

        public int Store(Span<float> store) => 0;
    }
}
