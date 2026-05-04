using SparseNeuralNetworkInferenceEngine.General;
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace SparseNeuralNetworkInferenceEngine.Engine
{
    public class ThreadPool : IThreadPool
    {
        protected record struct WorkItem
        {
            public Func<int, object> Func { get; init; }
            public Action<object> Done { get; init; }
            public Action Canceled { get; init; }
            public CancellationToken Ct { get; init; }

            public WorkItem(Func<int, object> func, Action<object> done, Action canceled, CancellationToken ct)
            {
                Func = func;
                Done = done;
                Ct = ct;
                Canceled = canceled;
            }
        }

        /// <inheritdoc/>
        public ThreadPriority Priority { get; init; }

        /// <inheritdoc/>
        public int Capacity { get; init; }

        /// <summary>
        /// The threads used to execute the Scheduled work.
        /// </summary>
        protected Thread[] threads;

        /// <summary>
        /// A semaphore to coordinate the tasks.
        /// </summary>
        protected SemaphoreSlim semaphore;

        /// <summary>
        /// A queue of work items to execute.
        /// </summary>
        protected ConcurrentQueue<WorkItem> workItems = new();

        /// <summary>
        /// The cancelation token to cancel all running threads and stop the threadPool.
        /// </summary>
        protected CancellationToken ct;

        /// <summary>
        /// Creating a new ThreadPool that can be used to shedule work
        /// </summary>
        /// <param name="threads">The number of threads to use.</param>
        /// <param name="capacity">The number of work items that may be scheduled.</param>
        /// <param name="priority">The priority of the threads to use.</param>
        public ThreadPool(int threads, int capacity, ThreadPriority priority = ThreadPriority.Normal, CancellationToken ct = default)
        {
            semaphore = new SemaphoreSlim(0, capacity);
            this.ct = ct;

            Priority = priority;
            this.threads = new Thread[threads];

            // Initialize Threads
            for (int i = 0; i < threads; i++)
            {
                int d = i;
                // Creating the new thread
                this.threads[i] = new Thread(() => BeginThread(d))
                {
                    Priority = priority,
                    IsBackground = true
                };

                this.threads[i].Start();
            }
        }

        /// <inheritdoc/>
        public Task<K>? Schedule<K>(Func<int,K> func, CancellationToken ct = default)
        {
            var tcs = new TaskCompletionSource<K>();

            workItems.Enqueue(new((i) => (object?)func(i)!, (object o) => tcs.SetResult((K)o), ()=>tcs.SetCanceled(ct), ct));

            semaphore.Release();

            return tcs.Task;
        }

        /// <summary>
        /// The threads entry point.
        /// </summary>
        /// <param name="threadId">The thread id that calls this Thread</param>
        protected void BeginThread(int threadId)
        {
            while (!ct.IsCancellationRequested)
            {
                // Waiting for work to be scheduled.
                var waitHandle =  semaphore.AvailableWaitHandle;
                WaitHandle.WaitAny([waitHandle, ct.WaitHandle]);

                if (ct.IsCancellationRequested)
                    break;

                // Try to retrieve a work item to work on.
                if (workItems.TryDequeue(out var workItem))
                {
                    if (workItem.Ct.IsCancellationRequested)
                    {
                        workItem.Canceled();
                        continue;
                    }

                    var result = workItem.Func.Invoke(threadId);

                    // Work has been completed, set the result for the associated task.
                    workItem.Done(result);
                }
            }
        }
    }
}
