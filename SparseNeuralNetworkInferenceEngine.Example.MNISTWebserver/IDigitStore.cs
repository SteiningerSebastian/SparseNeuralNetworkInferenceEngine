namespace SparseNeuralNetworkInferenceEngine.Example.MNISTWebserver
{
    public interface IDigitStore
    {
        public int[] DigitCoutner { get; }

        public void IncrementCounter(int digit);
    }
}
