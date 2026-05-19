using System;
using Gestalt.JsonCore;
class ML {
    

    static void Main()
    {
        int trainSample = 5000;
        int testSample = 1000;

        ModelCheckpoint modelCheckpoint = ModelCheckpoint.Load("mnist_784_128_128");
        NeuralNetwork network = modelCheckpoint.NeuralNetwork;
        Sequential sequential = network.Sequential;
        Optimizer optimizer = network.Optimizer;

        MNISTDataset train = MNISTLoader.Load("train", trainSample);
        MNISTDataset test = MNISTLoader.Load("t10k", testSample);

        var trainBatch = new Batch<MNISTSample>(train.Samples, trainSample);
        var testBatch = new Batch<MNISTSample>(test.Samples, testSample);

        modelCheckpoint.PrintStats();

        for(int i = 0; i < 10; i++)
        {
            int currentEpoch = modelCheckpoint.CurrentEpoch;

            train.Shuffle();

            while (train.HasNextBatch)
            {
                var miniBatch = train.GetBatch(32);
                
                network.Fit(miniBatch.X, miniBatch.Y);
            }

            Matrix testPred = sequential.Forward(testBatch.X);
            Matrix trainPred = sequential.Forward(trainBatch.X);

            double trainLoss = Loss.LossValue(trainPred, trainBatch.Y, network.ModelLoss);
            double testLoss = Loss.LossValue(testPred, testBatch.Y, network.ModelLoss);
            
            double trainAcc = Metrics.MultiClassificationAccuracy(trainPred, trainBatch.Y) * 100;
            double testAcc = Metrics.MultiClassificationAccuracy(testPred, testBatch.Y) * 100;
            
            sequential.Backward(Loss.LossGrad(trainPred, trainBatch.Y, network.ModelLoss));

            TrainingMonitor.LogEpoch(modelCheckpoint.CurrentEpoch, trainLoss, testLoss, trainAcc, testAcc, network);
            
            modelCheckpoint.Track(testLoss);

            modelCheckpoint.Track();
        }
    }
}