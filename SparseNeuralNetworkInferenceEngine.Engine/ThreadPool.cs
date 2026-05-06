using SparseNeuralNetworkInferenceEngine.General;
using System.Collections.Concurrent;
using System.Threading.Channels;

namespace SparseNeuralNetworkInferenceEngine.Engine
{
    public class ThreadPool : IThreadPool, IDisposable
    {
        protected record struct WorkItem
        {
            public Func<int, CancellationToken, object?> Func { get; init; }
            public Action<object?> Done { get; init; }
            public Action Canceled { get; init; }
            public CancellationToken CancelationToken { get; init; }

            public WorkItem(Func<int, CancellationToken, object?> func, Action<object?> done, Action canceled, CancellationToken ct)
            {
                Func = func;
                Done = done;
                CancelationToken = ct;
                Canceled = canceled;
            }
        }

        /// <inheritdoc/>
        public ThreadPriority Priority { get; init; }

        /// <inheritdoc/>
        public int Capacity { get; init; }

        public int NumberOfThreads => threads.Count;

        /// <summary>
        /// The threads used to execute the Scheduled work.
        /// </summary>
        protected IList<Thread> threads;

        /// <summary>
        /// A semaphore to coordinate the tasks.
        /// </summary>
        protected SemaphoreSlim semaphore;

        /// <summary>
        /// A queue of work items to execute.
        /// </summary>
        protected ConcurrentQueue<WorkItem> workItems = new();

        /// <summary>
        /// The cancelationTokenSource to be canceled when disposed.
        /// </summary>
        protected CancellationTokenSource cts;

        /// <summary>
        /// Creating a new ThreadPool that can be used to shedule work
        /// </summary>
        /// <param name="threads">The number of threads to use.</param>
        /// <param name="capacity">The number of work items that may be scheduled.</param>
        /// <param name="priority">The priority of the threads to use.</param>
        public ThreadPool(int threads, int capacity, ThreadPriority priority = ThreadPriority.Normal, CancellationTokenSource? cts = null)
        {
            if (cts is null)
            {
                cts = new CancellationTokenSource();
            }
            this.cts = cts;
            Capacity = capacity;
            semaphore = new SemaphoreSlim(0, capacity);

            Priority = priority;
            this.threads = new List<Thread>(threads);

            // Initialize Threads
            for (int i = 0; i < threads; i++)
            {
                int d = i;
                // Creating the new thread
                this.threads.Add(new Thread(() => BeginThread(d, cts.Token))
                {
                    Priority = priority,
                    IsBackground = true
                });

                this.threads[i].Start();
            }
        }

        /// <inheritdoc/>
        public Task<K>? Schedule<K>(Func<int, CancellationToken, K> func, CancellationToken ct = default)
        {
            if (workItems.Count >= Capacity)
                return null;

            ct = CancellationTokenSource.CreateLinkedTokenSource(ct, cts.Token).Token;

            var tcs = new TaskCompletionSource<K>();

            workItems.Enqueue(new((i, ct) => (object?)func(i, ct)!, (object? o) => tcs.SetResult((K)o!), () => tcs.SetCanceled(ct), ct));

            semaphore.Release();

            return tcs.Task;
        }

        public Task? Schedule(Action<int, CancellationToken> func, CancellationToken ct = default)
        {
            if (workItems.Count >= Capacity)
                return null;

            ct = CancellationTokenSource.CreateLinkedTokenSource(ct, cts.Token).Token;

            var tcs = new TaskCompletionSource();

            workItems.Enqueue(new((i, ct) => { func(i, ct); return null; }, (object? o) => tcs.SetResult(), () => tcs.SetCanceled(ct), ct));

            semaphore.Release();

            return tcs.Task;
        }

        /// <summary>
        /// The threads entry point.
        /// </summary>
        /// <param name="threadId">The thread id that calls this Thread</param>
        protected void BeginThread(int threadId, CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                // Waiting for work to be scheduled.
                var waitHandle = semaphore.AvailableWaitHandle;
                WaitHandle.WaitAny([waitHandle, ct.WaitHandle]);

                if (ct.IsCancellationRequested)
                    break;

                // Try to retrieve a work item to work on.
                if (workItems.TryDequeue(out var workItem))
                {
                    if (workItem.CancelationToken.IsCancellationRequested)
                    {
                        workItem.Canceled();
                        continue;
                    }

                    object? result = workItem.Func.Invoke(threadId, workItem.CancelationToken);

                    // Work has been completed, set the result for the associated task.
                    workItem.Done(result);
                }
            }
        }

        public void Dispose()
        {
            cts.Cancel(); // Cancel the current work.
            threads.Clear(); // Clear all threads
        }
    }
}
