using System;
using System.Diagnostics;
using Gestalt.JsonCore;
class ML {
    static void Main()
    {
        Stopwatch stopwatch = new Stopwatch();

        int trainSample = 5000;
        int testSample = 1000;

        ModelCheckpoint modelCheckpoint = ModelCheckpoint.Load("cnn-prototype-2");

        NeuralNetwork network = modelCheckpoint.NeuralNetwork;
        Sequential sequential = network.Sequential;
        Optimizer optimizer = network.Optimizer;

        network.Lambda = 0.0001;

        MNISTDataset train = MNISTLoader.Load("train", trainSample);
        MNISTDataset test = MNISTLoader.Load("t10k", testSample);

        var trainBatch = new Batch<MNISTSample>(train.Samples, 0, trainSample);
        var testBatch = new Batch<MNISTSample>(test.Samples, 0, testSample);

        modelCheckpoint.PrintStats();

        for(int i = 0; i < 1000; i++)
        {
            stopwatch.Start();

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


            TrainingMonitor.LogEpoch(currentEpoch, trainLoss, testLoss, trainAcc * 100, testAcc * 100, network, stopwatch);

            modelCheckpoint.Track(testLoss);

            modelCheckpoint.Track();
        }
    }
}