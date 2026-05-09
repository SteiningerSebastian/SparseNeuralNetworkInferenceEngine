namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface IHardwareAccelerator: ICloneable
    {
        /// <summary>
        /// Certain operartions need buffers, and the hardware accelerator is responsible for managing these buffers. 
        /// This method is called before inference to allow the hardware accelerator to prepare any necessary buffers based on the shape of the input tensor.
        ///
        /// This method may be called multiple times if the same hardware accelerator instance is used for multiple inferences with different input shapes. The hardware accelerator should manage its buffers accordingly, reusing them when possible and resizing or reallocating them when necessary.
        /// </summary>
        /// <param name="shape">The shape of the input tensor.</param>
        public void PrepareForInference(int[] shape);
    }
}
