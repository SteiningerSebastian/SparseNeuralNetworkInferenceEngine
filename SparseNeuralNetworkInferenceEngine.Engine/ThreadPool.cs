using SparseNeuralNetworkInferenceEngine.General;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace SparseNeuralNetworkInferenceEngine.Engine
{
    public unsafe sealed class ThreadPool : IThreadPool, IDisposable
    {
        private unsafe struct WorkItem
        {
            public delegate* managed<int, void*, void> Func { get; init; }
            public TaskCompletionSource TaskCompletionSource { get; init; }
            public void* Data { get; init; }

            public WorkItem(delegate* managed<int, void*, void> func, TaskCompletionSource tcs, void* data)
            {
                Func = func;
                TaskCompletionSource = tcs;
                Data = data;
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
        private IList<Thread> threads;

        /// <summary>
        /// A semaphore to coordinate the tasks.
        /// </summary>
        private SemaphoreSlim semaphore;

        /// <summary>
        /// A queue of work items to execute.
        /// </summary>
        private ConcurrentQueue<WorkItem> workItems = new();

        /// <summary>
        /// The cancelationTokenSource to be canceled when disposed.
        /// </summary>
        private CancellationTokenSource cts;

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
            semaphore = new SemaphoreSlim(0);

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Task? Schedule(delegate* managed<int, void*, void> func, void* data)
        {
            if (workItems.Count >= Capacity)
                return null;

            var tcs = new TaskCompletionSource();

            workItems.Enqueue(new(func, tcs, data));

            semaphore.Release();

            return tcs.Task;
        }

        /// <summary>
        /// The threads entry point.
        /// </summary>
        /// <param name="threadId">The thread id that calls this Thread</param>
        private void BeginThread(int threadId, CancellationToken ct)
        {
            CancellationTokenSource cts = CancellationTokenSource.CreateLinkedTokenSource(ct, this.cts.Token);
            ct = cts.Token;
            while (!ct.IsCancellationRequested)
            {
                for (int i = 0; i < 100; i++)
                {
                    // Try to retrieve a work item to work on.
                    if (workItems.TryDequeue(out var workItem))
                    {
                        workItem.Func(threadId, workItem.Data);

                        // Work has been completed, set the result for the associated task.
                        workItem.TaskCompletionSource.SetResult();
                    }
                    else
                    {
                        // No work item was found, wait a bit and try again.
                        // Up to 1ms wait time, then yield the thread to the OS scheduler to prevent busy waiting.
                        Thread.SpinWait(10);
                    }
                }

                semaphore.Wait(ct);
            }
        }

        public void Dispose()
        {
            cts.Cancel(); // Cancel the current work.
            threads.Clear(); // Clear all threads
        }
    }
}
