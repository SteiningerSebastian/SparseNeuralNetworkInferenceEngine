using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.HardwareAcceleration;
using SparseNeuralNetworkInferenceEngine.Math;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using SparseNeuralNetworkInferenceEngine.Model;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using ThreadPool = SparseNeuralNetworkInferenceEngine.Engine.ThreadPool;

Console.WriteLine("Hello, World!");

const int BATCH_SIZE = 64;

// 6 PCores that can do the AVX instructions fast.
IThreadPool threadPool = new ThreadPool(6, 1024, System.Threading.ThreadPriority.Highest);
IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
IInferenceEngine engine = new InferenceEngine(accelerator);

const int INPUT_SIZE = 784;

IModel model = new ModelSequential([
    new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(304, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(112, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, accelerator, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
    ], engine);

// Compile the model
model.Compile();

await model.LoadAsync($"E:/IUBScSNNIE/Models/Evaluation/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/model_parameters.bin");

float[] inps = await BinaryLoader.ReadFileToFloatEnumerableAsync($"E:/IUBScSNNIE/Models/Evaluation/MNIST_L_784_304_112_10_R_0e-0_ACC_98e-2_AVG_S_0/x_test_flattened.bin");

var inputLayout = new BatchValueTensorMemoryLayout(BATCH_SIZE, INPUT_SIZE);
var inputs = engine.AllocateUninitializedAlignedTensor<Tensor2D<float>, float>(inputLayout, BATCH_SIZE, INPUT_SIZE);
inputs.PopulateWithEnumerable(inps.AsSpan().Slice(0, BATCH_SIZE * INPUT_SIZE).ToArray());


float time = 0;
Stopwatch stopwatch = new Stopwatch();
for (int i = 1; i < 10000; i++)
{
    stopwatch.Restart();

    await model.InvokeAsync(inputs);

    stopwatch.Stop();

    time = time + 1 / (float)i * (stopwatch.ElapsedTicks * 1000000 / Stopwatch.Frequency - time);

    if (i % 1000 == 0)
    {
        Console.WriteLine("WarmUP average time: " + stopwatch.ElapsedTicks * 1000000 / Stopwatch.Frequency + " us | average: " + time);
    }
}

time = 0;
for (int i = 1; i < 100000; i++)
{
    stopwatch.Restart();

    await model.InvokeAsync(inputs);

    stopwatch.Stop();

    time = time + 1/(float)i*(stopwatch.ElapsedTicks * 1000000 / Stopwatch.Frequency - time);

    if (i % 1000 == 0)
    {
        Console.WriteLine("Current time: " + stopwatch.ElapsedTicks * 1000000/Stopwatch.Frequency + " us | average: " + time);
    }
}