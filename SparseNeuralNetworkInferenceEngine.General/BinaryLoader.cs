using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public static class BinaryLoader
    {
        /// <summary>
        /// Read the file to a float enumerable.
        /// </summary>
        /// <param name="path">The path to the file to read.</param>
        /// <returns>Returns an float array containing the data.</returns>
        public static async Task<float[]> ReadFileToFloatEnumerableAsync(string path)
        {
            byte[] parameters = await File.ReadAllBytesAsync(path);
            float[] par = new float[parameters.Length / sizeof(float)];
            for (int p = 0; p < par.Length; p++)
            {
                par[p] = BitConverter.ToSingle(parameters, p * sizeof(float));
            }
            return par;
        }

        /// <summary>
        /// Reat the file to a float enumerable.
        /// </summary>
        /// <param name="path">The path to the file to read.</param>
        /// <returns>Returns a float array containing the data.</returns>
        public static float[] ReadFileToFloatEnumerable(string path)
        {
            byte[] parameters = File.ReadAllBytes(path);
            float[] par = new float[parameters.Length / sizeof(float)];
            for (int p = 0; p < par.Length; p++)
            {
                par[p] = BitConverter.ToSingle(parameters, p * sizeof(float));
            }
            return par;
        }
    }
}
