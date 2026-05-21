using SparseNeuralNetworkInferenceEngine.General;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace SparseNeuralNetworkInferenceEngine.Engine
{
    public unsafe sealed class ThreadPool : IThreadPool, IDisposable
    {
        // Inspired by https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask
        [DllImport("kernel32.dll")]
        private static extern IntPtr SetThreadAffinityMask(IntPtr hThread, IntPtr dwThreadAffinityMask);

        // Inspired by https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentthread
        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentThread();

        const int MAX_SPIN_WAIT = 10000;

        private unsafe struct WorkItem
        {
            public delegate* managed<int, void*, void> Func { get; set; }
            public TaskCompletionSource TaskCompletionSource { get; set; }
            public void* Data { get; set; }

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
        /// A bag of free work items to avoid unnecessary allocations when scheduling work.
        /// </summary>
        private ConcurrentBag<WorkItem> workItemsBag = new();

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
                    IsBackground = true,
                });

                this.threads[i].Start();
            }

            // Pre-allocate work items to avoid unnecessary allocations during scheduling.
            for (int i = 0; i < capacity; i++)
            {
                workItemsBag.Add(new WorkItem());
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Task? Schedule(delegate* managed<int, void*, void> func, void* data)
        {
            if (workItems.Count >= Capacity)
                return null;

            var tcs = new TaskCompletionSource();

            if (workItemsBag.TryTake(out var item))
            {
                item.Func = func;
                item.Data = data;
                item.TaskCompletionSource = tcs;
                workItems.Enqueue(item);

                semaphore.Release();
            }
            else
            {
                throw new InvalidOperationException("Unable to schedule work item, no more capacity available.");
            }

            return tcs.Task;
        }

        /// <summary>
        /// The threads entry point.
        /// </summary>
        /// <param name="threadId">The thread id that calls this Thread</param>
        private void BeginThread(int threadId, CancellationToken ct)
        {
            // Pin the work to the performance cores to avoid unnecessary thread migrations and context switches.
            // And to ensure that they are performing scheduled work as equally as possible.
            Thread.BeginThreadAffinity();
            long mask = 1L << threadId;
            SetThreadAffinityMask(GetCurrentThread(), new IntPtr(mask));

            CancellationTokenSource cts = CancellationTokenSource.CreateLinkedTokenSource(ct, this.cts.Token);
            ct = cts.Token;
            WorkItem workItem;
            while (!ct.IsCancellationRequested)
            {
                for (int i = 0; i < MAX_SPIN_WAIT; i++)
                {
                    // Try to retrieve a work item to work on.
                    if (workItems.TryDequeue(out workItem))
                    {
                        // If the semaphore was acquired, we need to release it to avoid busy waiting.
                        semaphore.Wait(0);

                        workItem.Func(threadId, workItem.Data);

                        // Work has been completed, set the result for the associated task.
                        workItem.TaskCompletionSource.SetResult();

                        // Return the work item to the bag to be reused for future scheduled work.
                        workItemsBag.Add(workItem);

                        i = 0; // Reset the spin wait counter to avoid yielding the thread when there is still work to do.
                    }
                    Thread.SpinWait(1);
                }

                // Only yield the thread when there has not been any work for a while to avoid unnecessary context switches.
                semaphore.Wait(ct);

                if (workItems.TryDequeue(out workItem))
                {
                    workItem.Func(threadId, workItem.Data);

                    // Work has been completed, set the result for the associated task.
                    workItem.TaskCompletionSource.SetResult();

                    // Return the work item to the bag to be reused for future scheduled work.
                    workItemsBag.Add(workItem);
                }
                else
                {
                    // If we were released but someone else worked on it, we need to releas
                    // them as they after retriving the workItem also wait.
                    semaphore.Release();
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
