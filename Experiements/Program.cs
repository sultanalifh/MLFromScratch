using System;
using System.IO.Compression;

class ML {
    

    static void Main()
    {
        Sequential sequential = new Sequential();

        DenseLayer denseLayer = new DenseLayer(2,32);

        for(int i = 0; i < denseLayer.Weights.Data.Rows; i++)
        {
            for(int j = 0; j < denseLayer.Weights.Data.Cols; j++)
            {
                double z = Utility.RandomNormal();
                denseLayer.Weights.Data[i,j] = z * Math.Sqrt(2d / denseLayer.Weights.Data.Cols);
            }
        }

        sequential.Add(denseLayer);

        LayerNorm batchNormLayer = new LayerNorm(32,32);

        sequential.Add(batchNormLayer);

        Tanh reLULayer = new Tanh(32,32);
    
        sequential.Add(reLULayer);

        Dropout dropout1 = new Dropout(32, 32, 0.1);

        sequential.Add(dropout1);

        denseLayer = new DenseLayer(32,32);

        sequential.Add(denseLayer);

        for(int i = 0; i < denseLayer.Weights.Data.Rows; i++)
        {
            for(int j = 0; j < denseLayer.Weights.Data.Cols; j++)
            {
                double z = Utility.RandomNormal();
                denseLayer.Weights.Data[i,j] = z * Math.Sqrt(2d / denseLayer.Weights.Data.Cols);
            }
        }

        batchNormLayer = new LayerNorm(32,32);

        sequential.Add(batchNormLayer);

        reLULayer = new Tanh(32,32);
    
        sequential.Add(reLULayer);

        Dropout dropout2 = new Dropout(32, 32, 0.1);

        sequential.Add(dropout2);

        denseLayer = new DenseLayer(32,3);

        sequential.Add(denseLayer);

        Softmax softmax = new Softmax(3,3);

        sequential.Add(softmax);

        // Momentum optimizer = new Momentum(sequential, 0.01, 0.9);
        // SGD optimizer = new SGD(sequential, 0.01);
        Adam optimizer = new Adam(sequential, 0.01);

        SpiralDataset train = new SpiralDataset(3, 100);
        SpiralDataset test = new SpiralDataset(3, 100);
        train.Generate();

        for(int i = 0; i < 300; i++)
        {
            dropout1.IsTraining = dropout2.IsTraining = (i + 1) % 10 != 0;

            optimizer.ZeroGrad();
            Matrix pred = sequential.Predict(train.X);
            sequential.Backward(pred, train.Y);
            optimizer.Step();
            
            if((i + 1) % 10 == 0)
            {
                test.Generate();
                
                int epoch = i + 1;
                double loss = Loss.CrossEntropyValue(pred, train.Y);
                double trainAcc = Metrics.MultiClassificationAccuracy(pred, train.Y) * 100;

                optimizer.ZeroGrad();
                pred = sequential.Predict(test.X);
                double testAcc = Metrics.MultiClassificationAccuracy(pred, test.Y) * 100;
                sequential.Backward(pred, test.Y);

                TrainingMonitor.LogEpoch(epoch, loss, trainAcc, testAcc, sequential);

                train.Generate();
            }

            // if((i + 1) % 50 == 0)
            // {
            //     DecisionBoundary.SaveSpiralDatasetBoundaryImage(sequential, "L2_epoch_" + (i + 1) + ".png");
            // }
        }

        // TrainingMonitor.CheckNumericalGrad(sequential, train.X, train.Y, 0);

        // DecisionBoundary.SaveSpiralDatasetBoundaryImage(sequential, "output.jpeg");
    }
}