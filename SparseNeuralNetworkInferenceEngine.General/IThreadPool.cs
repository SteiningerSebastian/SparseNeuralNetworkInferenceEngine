using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;

namespace SparseNeuralNetworkInferenceEngine.General
{
    public unsafe interface IThreadPool
    {
        public ThreadPriority Priority { get; }

        public int Capacity { get; }

        public int NumberOfThreads  { get; }

        public delegate void ParallelLoop(int threadId, int index);

        /// <summary>
        /// Schedules new work to be done.
        /// </summary>
        /// <param name="func">The function to work on.</param>
        /// <param name="data">The data to pass to the function.</param>
        /// <returns>Null is returned if work could not be scheduled, else the Task for the work is returned.</returns>
        public Task? Schedule(delegate* managed<int, void*, void> func, void* data);
     }
}
