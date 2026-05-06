using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public interface IThreadPool
    {
        public ThreadPriority Priority { get; }

        public int Capacity { get; }

        public int NumberOfThreads  { get; }

        public delegate void ParallelLoop(int threadId, int index);

        /// <summary>
        /// Schedules new work to be done.
        /// </summary>
        /// <param name="func">The function to work on.</param>
        /// <param name="cts">The cacnelation token to cancel the work.</param>
        /// <returns>Null is returned if work could not be scheduled, else the Task for the work is returned.</returns>
        public Task<K>? Schedule<K>(Func<int, CancellationToken, K> func, CancellationToken cts = default);

        /// <summary>
        /// Schedules new work to be done.
        /// </summary>
        /// <param name="func">The function to work on.</param>
        /// <param name="cts">The cacnelation token to cancel the work.</param>
        /// <returns>Null is returned if work could not be scheduled, else the Task for the work is returned.</returns>
        public Task? Schedule(Action<int, CancellationToken> func, CancellationToken cts = default);
     }
}
