using System;
using System.Collections.Generic;
using System.Text;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface IThreadPool
    {
        public ThreadPriority Priority { get; }

        public int Capacity { get; }

        /// <summary>
        /// Schedules new work to be done.
        /// </summary>
        /// <param name="func">The function to work on.</param>
        /// <param name="cts">The cacnelation token to cancel the work.</param>
        /// <returns>Null is returned if work could not be scheduled, else the Task for the work is returned.</returns>
        public Task<K>? Schedule<K>(Func<int, K> func, CancellationToken cts = default);

    }
}
