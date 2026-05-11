using System;

class ML {
    

    static void Main()
    {
        Sequential sequential = new Sequential();

        DenseLayer denseLayer = new DenseLayer(2,32);

        for(int i = 0; i < denseLayer.Weights.Data.Rows; i++)
        {
            for(int j = 0; j < denseLayer.Weights.Data.Cols; j++)
            {
                double u1 = Utility.Random.NextDouble();
                double u2 = Utility.Random.NextDouble();
                double z = Utility.RandomNormal();
                denseLayer.Weights.Data[i,j] = z * Math.Sqrt(2d / denseLayer.Weights.Data.Cols);
            }
        }

        sequential.Add(denseLayer);

        LayerNorm batchNormLayer = new LayerNorm(32,32);

        sequential.Add(batchNormLayer);

        ReLU reLULayer = new ReLU(32,32);
    
        sequential.Add(reLULayer);

        denseLayer = new DenseLayer(32,32);

        sequential.Add(denseLayer);

        for(int i = 0; i < denseLayer.Weights.Data.Rows; i++)
        {
            for(int j = 0; j < denseLayer.Weights.Data.Cols; j++)
            {
                double u1 = Utility.Random.NextDouble();
                double u2 = Utility.Random.NextDouble();
                double z = Utility.RandomNormal();
                denseLayer.Weights.Data[i,j] = z * Math.Sqrt(2d / denseLayer.Weights.Data.Cols);
            }
        }

        batchNormLayer = new LayerNorm(32,32);

        sequential.Add(batchNormLayer);

        reLULayer = new ReLU(32,32);
    
        sequential.Add(reLULayer);

        denseLayer = new DenseLayer(32,3);

        sequential.Add(denseLayer);

        Softmax softmax = new Softmax(3,3);

        sequential.Add(softmax);

        // Momentum optimizer = new Momentum(sequential, 0.01, 0.9);
        // SGD optimizer = new SGD(sequential, 0.01);
        Adam optimizer = new Adam(sequential, 0.01);

        SpiralDataset dataset = new SpiralDataset(3, 100);
        dataset.Generate();

        for(int i = 0; i < 300; i++)
        {
            optimizer.ZeroGrad();
            Matrix pred = sequential.Predict(dataset.X);
            sequential.Backward(pred, dataset.Y);
            optimizer.Step();
        }

        TrainingMonitor.CheckNumericalGrad(sequential, dataset.X, dataset.Y, 0);
    }
}