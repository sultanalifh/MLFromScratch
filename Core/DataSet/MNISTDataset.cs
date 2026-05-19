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
}