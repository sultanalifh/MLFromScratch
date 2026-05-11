static class TrainingMonitor
{
    public static void LogEpoch(int epoch, double loss, double trainAcc, double testAcc, Sequential model)
    {
        Console.WriteLine($"--- Epoch {epoch} ---");
        Console.WriteLine($"Loss: {loss}");
        Console.WriteLine($"Train Accuracy: {trainAcc}%");
        Console.WriteLine($"Test Accuracy: {testAcc}%");

        Console.WriteLine("--- Layer Stats --- ");

        int numLayers = model.Layers.Count;

        for(int i = 0; i < numLayers; i++)
        {
            Layer layer = model.Layers[i];

            string layerName = layer.GetType().Name;

            Console.WriteLine($"Layer {i + 1}, {layerName}");
            Console.WriteLine($"Input size: {layer.InputSize}");
            Console.WriteLine($"Output size: {layer.OutputSize}");
            
            if(layer.CachedOutput != null)
            {
                Console.WriteLine($"Activation mean: {Stats.Mean(layer.CachedOutput)}");
                Console.WriteLine($"Activation var: {Stats.Variance(layer.CachedOutput)}");

                if(layer is ReLU)
                {
                    Console.WriteLine($"Zero ratio: {Stats.ZeroRatio(layer.CachedOutput)}");
                }

                if (Stats.HasNaN(layer.CachedOutput))
                {
                    Console.WriteLine("Warning: NaN detected!");
                }
            }

            foreach(Parameter parameter in layer.Parameters())
            {
                Console.WriteLine();
                Console.WriteLine($"{parameter.Name} mean: {Stats.Mean(parameter.Data)}");
                Console.WriteLine($"{parameter.Name} var: {Stats.Variance(parameter.Data)}");
                Console.WriteLine($"{parameter.Name} Grad mean: {Stats.MeanAbs(parameter.Grad)}");
                Console.WriteLine($"{parameter.Name} Grad max: {Stats.MaxAbs(parameter.Grad)}");

                if (Stats.HasNaN(parameter.Data) || Stats.HasNaN(parameter.Grad))
                {
                    Console.WriteLine("Warning: NaN detected!");
                }
            }

            Console.WriteLine();
        }
    }

    public static void CheckNumericalGrad(Sequential sequential, Matrix testcase, Matrix ans, int layerPos)
    {
        if(layerPos < 0 || layerPos >= sequential.Layers.Count)
        {
            throw new ArgumentOutOfRangeException("Index layer out of bounds!");
        }

        Layer layer = sequential.Layers[layerPos];
        string layerName = layer.GetType().Name;

        Console.WriteLine($"--- Numerical Grad analysis on layer {layerPos}, {layerName} ---");

        foreach(Parameter parameter in layer.Parameters())
        {
            sequential.ZeroGrad();

            Console.WriteLine();
            Console.WriteLine($"--- Gradient Check for {parameter.Name} ---");

            double originalWeight = parameter.Data[0,0];

            sequential.Backward(sequential.Predict(testcase), ans);

            double analyticalGrad = parameter.Grad[0,0];

            parameter.Data[0,0] = originalWeight + Utility.e5Eps;
            
            Matrix sample1 = sequential.Predict(testcase);
            double sample1Loss = Loss.CrossEntropyValue(sample1, ans);

            parameter.Data[0,0] = originalWeight - Utility.e5Eps;

            Matrix sample2 = sequential.Predict(testcase);
            double sample2Loss = Loss.CrossEntropyValue(sample2, ans);

            double numericalGrad = (sample1Loss - sample2Loss)/(2*Utility.e5Eps);

            double diff = Math.Abs(analyticalGrad - numericalGrad);
            double accuracy = (1 - diff) * 100;

            Console.WriteLine($"Analytical Gradient: {analyticalGrad}");
            Console.WriteLine($"Numerical Gradient: {numericalGrad}");
            Console.WriteLine($"Accuracy: {accuracy}%");

            parameter.Data[0,0] = originalWeight;
        }
    }
}