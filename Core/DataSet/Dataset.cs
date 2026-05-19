abstract class Dataset<TSample> where TSample : Sample
{
    protected int Iterator;
    public List<TSample> Samples = new List<TSample>();
    public bool HasNextBatch => Iterator < Samples.Count;
    
    public Dataset()
    {
        
    }

    public Dataset(List<TSample> samples)
    {
        Samples = samples;
    }
    public virtual void Shuffle()
    {
        Samples = Samples.OrderBy(sample => Utility.Random.NextDouble()).ToList();

        Iterator = 0;
    }

    public abstract Batch<TSample> GetBatch(int numBatch);
}