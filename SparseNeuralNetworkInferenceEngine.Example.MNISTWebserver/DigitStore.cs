namespace SparseNeuralNetworkInferenceEngine.Example.MNISTWebserver
{
    /// <summary>
    /// A very simple in-memory store for counting the number of times each digit (0-9) has been predicted by the model. This class implements the IDigitStore interface, which defines a method for incrementing the counter for a given digit. The counters are stored in an array of integers, where the index corresponds to the digit and the value at that index represents the count for that digit.
    /// </summary>
    public class DigitStore : IDigitStore
    {
        public DigitStore(int digits)
        {
            DigitCoutner = new int[digits];
        }

        public int[] DigitCoutner { get; init; }

        public void IncrementCounter(int digit)
        {
            if (digit >= 0 && digit < 10)
            {
                DigitCoutner[digit]++;
            }
        }
    }
}
