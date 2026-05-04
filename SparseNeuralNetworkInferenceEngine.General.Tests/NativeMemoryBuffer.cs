namespace SparseNeuralNetworkInferenceEngine.General.Tests
{
    public class NativeMemoryBuffer
    {
        [Fact]
        public void Constructor()
        {
            NativeMemoryBufferManager<float> manager = new NativeMemoryBufferManager<float>(-1);

            {
                var buffer = manager.GetBuffer(16, sizeof(float));
                buffer.Buffer[3] = 3;
                buffer.Buffer[6] = 5;
                buffer.Dispose();
            } // here the buffer should be disposed. (call because GC may not have ran)

            // We expect the same buffer
            var buffer1 = manager.GetBuffer(16, sizeof(float));
            Assert.Equal(3, buffer1.Buffer[3]);
            Assert.Equal(5, buffer1.Buffer[6]);
        }

        [Fact]
        public void TestMultipleBuffers()
        {
            NativeMemoryBufferManager<float> manager = new NativeMemoryBufferManager<float>(-1);

            {
                var buffer = manager.GetBuffer(16, sizeof(float));
                buffer.Buffer[3] = 3;
                buffer.Buffer[6] = 5;
                buffer.Dispose();
            } // here the buffer should be disposed. (call because GC may not have ran)

            {
                var buffer = manager.GetBuffer(34, sizeof(float));
                buffer.Buffer[3] = 3;
                buffer.Buffer[6] = 5;
                buffer.Dispose();
            } // here the buffer should be disposed. (call because GC may not have ran)

            // We expect the same buffer
            var buffer1 = manager.GetBuffer(13, sizeof(float));
            Assert.Equal(3, buffer1.Buffer[3]);
            Assert.Equal(5, buffer1.Buffer[6]);
            buffer1.Dispose();

            var buffer2 = manager.GetBuffer(13, sizeof(float));
            var buffer3 = manager.GetBuffer(13, sizeof(float));

            var buffer4 = manager.GetBuffer(34, sizeof(float));

            Assert.Equal(3, buffer4.Buffer[3]);
            Assert.Equal(5, buffer4.Buffer[6]);
        }


        [Fact]
        public void TestMultipleLargeBuffers()
        {
            NativeMemoryBufferManager<float> manager = new NativeMemoryBufferManager<float>(-1);

            {
                var buffer = manager.GetBuffer(1245274, sizeof(float));
                buffer.Buffer[3] = 3;
                buffer.Buffer[6] = 5;
                buffer.Dispose();
            } // here the buffer should be disposed. (call because GC may not have ran)

            {
                var buffer = manager.GetBuffer(4668843, sizeof(float));
                buffer.Buffer[3] = 4;
                buffer.Buffer[6] = 6;
                buffer.Dispose();
            } // here the buffer should be disposed. (call because GC may not have ran)

            // We expect the same buffer
            var buffer1 = manager.GetBuffer(1245274, sizeof(float));

            Assert.Equal(1245274, buffer1.Buffer.Length);

            Assert.Equal(3, buffer1.Buffer[3]);
            Assert.Equal(5, buffer1.Buffer[6]);
            buffer1.Dispose();

            var buffer2 = manager.GetBuffer(1245274, sizeof(float));
            var buffer3 = manager.GetBuffer(1245274, sizeof(float));

            var buffer4 = manager.GetBuffer(4668843, sizeof(float));
            Assert.Equal(4668843, buffer4.Buffer.Length);


            Assert.Equal(4, buffer4.Buffer[3]);
            Assert.Equal(6, buffer4.Buffer[6]);
        }
    }
}
