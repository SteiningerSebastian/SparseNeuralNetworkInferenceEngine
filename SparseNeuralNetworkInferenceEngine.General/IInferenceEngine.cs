using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface IInferenceEngine
    {
        public IThreadPool ThreadPool { get; }

    }
}
