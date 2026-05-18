static class TrainingMonitor
{
    public static void LogEpoch(int epoch, double loss, double trainAcc, double testAcc, NeuralNetwork network)
    {
        Console.WriteLine($"--- Epoch {epoch} ---");
        Console.WriteLine($"Loss: {loss}");
        Console.WriteLine($"Train Accuracy: {trainAcc}%");
        Console.WriteLine($"Test Accuracy: {testAcc}%");
        Console.WriteLine();
        Console.WriteLine("--- Layer Stats --- ");

        Sequential model = network.Sequential;

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

                if(layer is ReLU || layer is LeakyReLU)
                {
                    Console.WriteLine($"Zero ratio: {Stats.ZeroRatio(layer.CachedOutput) * 100}%");
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

    public static void CheckNumericalGrad(NeuralNetwork network, Matrix testcase, Matrix ans, int layerPos)
    {
        Sequential sequential = network.Sequential;
        Optimizer optimizer = network.Optimizer;


        if(layerPos < 0 || layerPos >= sequential.Layers.Count)
        {
            throw new ArgumentOutOfRangeException("Index layer out of bounds!");
        }
        else if(testcase.Rows != ans.Rows)
        {
            throw new ArgumentException("Shape mismatch!");
        }

        Layer layer = sequential.Layers[layerPos];
        int batchSize = testcase.Rows;
        string layerName = layer.GetType().Name;

        Console.WriteLine($"--- Numerical Grad analysis on layer {layerPos}, {layerName} ---");

        foreach(Parameter parameter in layer.Parameters())
        {
            optimizer.ZeroGrad();

            Console.WriteLine();
            Console.WriteLine($"--- Gradient Check for {parameter.Name} ---");

            double originalWeight = parameter.Data[0,0];
            Matrix yGrad = Loss.LossGrad(sequential.Forward(testcase), ans, network.ModelLoss);
            sequential.Backward(yGrad);

            double analyticalGrad = parameter.Grad[0,0];

            parameter.Data[0,0] = originalWeight + Utility.e5Eps;
            
            Matrix sample1 = sequential.Forward(testcase);
            double sample1Loss = Loss.LossValue(sample1, ans, network.ModelLoss) * batchSize;

            parameter.Data[0,0] = originalWeight - Utility.e5Eps;

            Matrix sample2 = sequential.Forward(testcase);
            double sample2Loss = Loss.LossValue(sample2, ans, network.ModelLoss) * batchSize;

            double numericalGrad = (sample1Loss - sample2Loss) / (2 * Utility.e5Eps);

            double diff = Math.Abs(analyticalGrad - numericalGrad);
            double accuracy = diff / (Math.Abs(analyticalGrad) + Math.Abs(numericalGrad) + 1e-8);

            Console.WriteLine($"Analytical Gradient: {analyticalGrad}");
            Console.WriteLine($"Numerical Gradient: {numericalGrad}");
            Console.WriteLine($"Relative Error: {accuracy}");

            parameter.Data[0,0] = originalWeight;
        }
    }
}