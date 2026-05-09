using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using ThreadPool = SparseNeuralNetworkInferenceEngine.Engine.ThreadPool;

namespace SparseNeuralNetworkInferenceEninge.Engine.Tests
{
    public class ThreadPoolTests
    {
        protected unsafe struct DataWork
        {
            public bool* worked;
        }

        public unsafe static void Work(int threadId, void* data)
        {
            *(*(DataWork*)data).worked = true;
        }

        [Fact]
        public unsafe void Constructor()
        {
            CancellationTokenSource cts = new CancellationTokenSource(1000);

            IThreadPool threadPool = new ThreadPool(4, 100, ThreadPriority.Normal, cts);

            bool worked = false;

            var dataWork = new DataWork()
            {
                worked = &worked
            };

            var task = threadPool.Schedule(&Work, &dataWork);
            Assert.True(task != null);
            task.Wait();
            Assert.True(worked);
        }

        protected unsafe struct DataWorkList
        {
            public bool* worked;
        }

        public unsafe static void WorkList(int threadId, void* data)
        {
            bool* worked = (*(DataWorkList*)data).worked + threadId;
            Thread.Sleep(100);
            *worked = true;
        }

        [Fact]
        public unsafe void TestMultipleInvokations()
        {
            CancellationTokenSource cts = new CancellationTokenSource(3000);

            IThreadPool threadPool = new ThreadPool(4, 10, ThreadPriority.Normal, cts);

            bool[] threadsWorked = new bool[4];
            var handl = GCHandle.Alloc(threadsWorked, GCHandleType.Pinned);

            var tasks = new List<Task>();

            var dataList = new DataWorkList()
            {
                worked = (bool*)handl.AddrOfPinnedObject()
            };

            for (int i = 0; i < 10; i++)
            {
                // Ignore the task that returns the result.
                var t = threadPool.Schedule(&WorkList, &dataList);

                Assert.NotNull(t);

                tasks.Add(t);
            }

            Task.WhenAll(tasks).Wait();

            // Test if every thread did some work
            Assert.True(threadsWorked[0]);
            Assert.True(threadsWorked[1]);
            Assert.True(threadsWorked[2]);
            Assert.True(threadsWorked[3]);
        }
    }
}
