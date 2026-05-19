using SparseNeuralNetworkInferenceEngine.Engine;
using SparseNeuralNetworkInferenceEngine.Example.MNISTWebserver;
using SparseNeuralNetworkInferenceEngine.General;
using SparseNeuralNetworkInferenceEngine.HardwareAcceleration;
using SparseNeuralNetworkInferenceEngine.Math;
using SparseNeuralNetworkInferenceEngine.Math.Tensor;
using SparseNeuralNetworkInferenceEngine.Model;
using ThreadPool = SparseNeuralNetworkInferenceEngine.Engine.ThreadPool;

var builder = WebApplication.CreateBuilder(args);

Thread.CurrentThread.Priority = ThreadPriority.Highest;

IThreadPool threadPool = new ThreadPool(6, 1024, ThreadPriority.BelowNormal);
IHardwareAccelerator accelerator = new AVXHardwareAccelerator(threadPool);
IInferenceEngine engine = new InferenceEngine(accelerator);

// For simplicity we use a batch size of 1 for the webserver, but the model can be compiled with any batch size and input size as long as they are multiples of 16.
// In a production environment you would want to use a larger batch size to maximize throughput, but for this example we want to minimize latency and make it easier to test with single inputs.
const int BATCH_SIZE = 16;
const int INPUT_SIZE = 784;

ModelSequential model = new ModelSequential([
    new InputLayer([BATCH_SIZE, INPUT_SIZE]),
                new DenseLayerAvx(304, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(112, threadPool.NumberOfThreads, accelerator),
                new DenseLayerAvx(16, threadPool.NumberOfThreads, accelerator, false),
                new ActivationLayer(new SoftMaxActivationFunction(10)),
                new OutputLayer(10)
    ], engine);

// Compile the model
model.Compile();

await model.LoadAsync("E:/IUBScSNNIE/Models/MNIST304_112_10_Sparsity_59/model_parameters.bin");

builder.Services.AddSingleton<IModel>(model);
builder.Services.AddSingleton<IInferenceEngine>(engine);
builder.Services.AddSingleton<IDigitStore>(new DigitStore(10));

builder.Services.AddSignalR();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapHub<MNISTHub>("/mnist");

app.Run();
