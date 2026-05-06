using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using System.Collections.Concurrent;
using ThreadPool = SparseNeuralNetworkInferenceEngine.Engine.ThreadPool;

namespace SparseNeuralNetworkInferenceEninge.Engine.Tests
{
    public class ThreadPoolTests
    {
        [Fact]
        public async Task Constructor()
        {
            CancellationTokenSource cts = new CancellationTokenSource(1000);

            IThreadPool threadPool = new ThreadPool(4, 100, ThreadPriority.Normal, cts);

            bool worked = false;

            var task = threadPool.Schedule((i, ct) => { worked = true; return true; });

            Assert.True(task != null);
            Assert.True(await task);
            Assert.True(worked);
        }

        [Fact]
        public async Task CheckThreadPriority()
        {
            CancellationTokenSource cts = new CancellationTokenSource(1000);

            IThreadPool threadPool = new ThreadPool(4, 100, ThreadPriority.BelowNormal, cts);
            ThreadPriority setPriority = ThreadPriority.Normal;
            var task = threadPool.Schedule((i, ct) =>
            {
                setPriority = Thread.CurrentThread.Priority;
                return true;
            });

            Assert.NotNull(task);

            await task;

            Assert.Equal(ThreadPriority.BelowNormal, setPriority);
        }

        [Fact]
        public async Task TestMultipleInvokations()
        {
            CancellationTokenSource cts = new CancellationTokenSource(3000);

            IThreadPool threadPool = new ThreadPool(4, 10, ThreadPriority.Normal, cts);

            ConcurrentBag<int> threadsWorked = new();

            var tasks = new List<Task<bool>>();

            for (int i = 0; i < 10; i++)
            {
                // Ignore the task that returns the result.
                var t = threadPool.Schedule<bool>((i, ct) =>
                 {
                     threadsWorked.Add(i);
                     Thread.Sleep(1000); 
                     return true;
                 });

                Assert.NotNull(t);

                tasks.Add(t);
            }

            await Task.WhenAll(tasks);

            // Test if every thread did some work
            Assert.Contains(0, threadsWorked);
            Assert.Contains(1, threadsWorked);
            Assert.Contains(2, threadsWorked);
            Assert.Contains(3, threadsWorked);
        }

        [Fact]
        public async Task TestCancel()
        {
            CancellationTokenSource cts = new CancellationTokenSource(3000);

            IThreadPool threadPool = new ThreadPool(4, 100, ThreadPriority.Normal, cts);

            ConcurrentBag<int> threadsWorked = new();
            ConcurrentBag<int> tasksWorked = new();

            var tasks = new List<Task<bool>>();

            for (int i = 0; i < 10; i++)
            {
                CancellationTokenSource taskCts = new();
                int d = i;

                // Ignore the task that returns the result.
                var t = threadPool.Schedule<bool>((n, ct) =>
                {
                    threadsWorked.Add(n);
                    tasksWorked.Add(d);
                    Thread.Sleep(10);
                    return true;
                }, taskCts.Token);

                Assert.NotNull(t);

                if (d == 7)
                {
                    taskCts.Cancel(); // Dont work this.
                }

                tasks.Add(t);
            }


            await Assert.ThrowsAsync<TaskCanceledException>(async () => await Task.WhenAll(tasks));

            Assert.Equal(9, tasksWorked.Count);

            // Test if every thread did some work
            Assert.Contains(0, tasksWorked);
            Assert.Contains(1, tasksWorked);
            Assert.Contains(2, tasksWorked);
            Assert.Contains(3, tasksWorked);
            Assert.Contains(4, tasksWorked);
            Assert.Contains(5, tasksWorked);
            Assert.Contains(6, tasksWorked);
            Assert.Contains(8, tasksWorked);
            Assert.Contains(9, tasksWorked);

            // Making sure the canceled task is not worked on.
            Assert.DoesNotContain(7, tasksWorked);
        }
    }
}
