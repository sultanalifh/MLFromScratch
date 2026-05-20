class MNISTDataset : Dataset<MNISTSample>
{
    public override Batch<MNISTSample> GetBatch(int numBatch)
    {
        numBatch = Math.Min(Iterator + numBatch, Samples.Count - Iterator);

        if(numBatch <= 0)
        {
            throw new ArgumentException("Invalid number of batch of " + numBatch);
        }

        List<MNISTSample> sample_batch = Samples.Skip(Iterator).Take(numBatch).ToList();

        Batch<MNISTSample> batch = new Batch<MNISTSample>(sample_batch, numBatch);

        Iterator += numBatch;

        return batch;
    }


    public static int[] GetConfusion(NeuralNetwork network, MNISTDataset dataset)
    {
        Sequential sequential = network.Sequential;

        int[] digit = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
        int numSample = dataset.Samples.Count;

        var samples = new Batch<MNISTSample>(dataset.Samples, numSample);

        Matrix yPred = sequential.Forward(samples.X);
        
        for(int i = 0; i < numSample; i++)
        {
            int maxProb = 0;

            for(int j = 0; j < 10; j++)
            {
                if(yPred[i,j] > yPred[i, maxProb])
                {
                    maxProb = j;
                }
            }

            if(digit[maxProb] == -1 || yPred[digit[maxProb], maxProb] < yPred[i, maxProb])
            {
                digit[maxProb] = i;
            }
        }

        return digit;
    }

    public static List<int> GetError(NeuralNetwork network, MNISTDataset dataset)
    {
        List<int> error = new List<int>();

        Sequential sequential = network.Sequential;

        int numSample = dataset.Samples.Count;
        var samples = new Batch<MNISTSample>(dataset.Samples, numSample);

        Matrix yPred = sequential.Forward(samples.X);

        for(int i = 0; i < numSample; i++)
        {
            int maxProb = 0;
            int label = dataset.Samples[i].Label;

            for(int j = 0; j < 10; j++)
            {
                if(yPred[i,j] > yPred[i, maxProb])
                {
                    maxProb = j;
                }
            }

            if(label != maxProb)
            {
                error.Add(i);
            }
        }

        return error;
    }
}