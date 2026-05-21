using System;
using Gestalt.JsonCore;
class ML {
    static void Main()
    {
        int trainSample = 10;
        int testSample = 100;

        ModelCheckpoint modelCheckpoint = ModelCheckpoint.Load("test");

        NeuralNetwork network = modelCheckpoint.NeuralNetwork;
        Sequential sequential = network.Sequential;
        Optimizer optimizer = network.Optimizer;

        MNISTDataset train = MNISTLoader.Load("train", trainSample);
        MNISTDataset test = MNISTLoader.Load("t10k", testSample);

        var trainBatch = new Batch<MNISTSample>(train.Samples, trainSample);
        var testBatch = new Batch<MNISTSample>(test.Samples, testSample);

        modelCheckpoint.PrintStats();

        for(int i = 0; i < 1; i++)
        {
            int currentEpoch = modelCheckpoint.CurrentEpoch;

            train.Shuffle();

            while (train.HasNextBatch)
            {
                var miniBatch = train.GetBatch(32);

                network.Fit(miniBatch.X, miniBatch.Y);
            }

            
            Tensor testPred = sequential.Forward(testBatch.X);
            Tensor trainPred = sequential.Forward(trainBatch.X);

            Tensor trainGrad = Loss.LossGrad(trainPred, trainBatch.Y, network.ModelLoss);

            double testLoss = Loss.LossValue(testPred, testBatch.Y, network.ModelLoss);
            double trainLoss = Loss.LossValue(trainPred, trainBatch.Y, network.ModelLoss);

            double testAcc = Metrics.MultiClassificationAccuracy(testPred, testBatch.Y);
            double trainAcc = Metrics.MultiClassificationAccuracy(trainPred, trainBatch.Y);

            sequential.Backward(trainGrad);


            TrainingMonitor.LogEpoch(currentEpoch, trainLoss, testLoss, trainAcc * 100, testAcc * 100, network);

            modelCheckpoint.Track(testLoss);

            modelCheckpoint.Track();
        }

        // for(int i = sequential.Layers.Count - 1; i >= 0; i--)
        // {
        //     TrainingMonitor.CheckNumericalGrad(network, testBatch.X, testBatch.Y, i);
        // }
    }
}