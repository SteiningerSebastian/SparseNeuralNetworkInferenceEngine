using Math.Tensor;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

Console.WriteLine("Hello, World!");

Tensor1D<float> myTensor = new Tensor1D<float>(100);
for(int i = 0; i < 100; i++)
{
    myTensor[i] = i;
}

Console.WriteLine(string.Join(',', myTensor));