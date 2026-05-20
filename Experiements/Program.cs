using System;
using Gestalt.JsonCore;
class ML {
    

    static void Main()
    {
        Tensor tensor = new Tensor([3,3,3]);

        Console.WriteLine(tensor.GetFlatIndex([2,1,2]));
    }
}