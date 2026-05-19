class Batch<TSample> where TSample : Sample
{
    public int Size;
    public Matrix X;

    public Matrix Y;

    public Batch(List<TSample> samples, int size)
    {
        Size = size;

        int inputSize = samples[0].Input.Length;
        int targetSize = samples[0].Target.Length;

        X = new Matrix(size, inputSize);
        Y = new Matrix(size, targetSize);

        for(int i = 0; i < size; i++)
        {
            Sample sample = samples[i];
            double[] input = sample.Input;
            double[] target = sample.Target;

            for(int j = 0; j < inputSize; j++)
            {
                X[i,j] = input[j];
            }

            for(int j = 0; j < targetSize; j++)
            {
                Y[i,j] = target[j];
            }
        }
    }
}