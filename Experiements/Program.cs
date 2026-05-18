using System;
using Gestalt.JsonCore;
class ML {
    

    static void Main()
    {

        // ModelCheckpoint modelCheckpoint = ModelCheckpoint.Load("spiral_dataset.model");
        // NeuralNetwork network = modelCheckpoint.NeuralNetwork;
        // Sequential sequential = network.Sequential;
        // Optimizer optimizer = network.Optimizer;

        // modelCheckpoint.PrintStats();

        // SpiralDataset train = new SpiralDataset(3, 50);
        // SpiralDataset test = new SpiralDataset(3, 300);

        // DecisionBoundary.SaveSpiralDatasetBoundaryImage(network, "spiral.jpeg");

        // for(int i = 0; i < 30000; i++)
        // {
        //     network.Fit(train.X, train.Y);
        //     modelCheckpoint.Track();

        //     if((i + 1) % 50 == 0)
        //     {
        //         test.Generate();

        //         Matrix yPred = network.Predict(test.X);
        //         double loss = Loss.LossValue(yPred, test.Y, network.ModelLoss);
        //         modelCheckpoint.Track(loss);

        //         train.Generate();
        //     }
        // }

    }
}