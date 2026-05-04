using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNEuronalNetworkInferenceEngine.Engine
{
    internal interface IInferenceEngine
    {
        public IThreadPool ThreadPool { get; }

    }
}
