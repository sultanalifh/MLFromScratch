class Dropout : Layer
{
    public double DropoutRate;

    public Tensor Mask;
    public bool IsTraining;


    public Dropout(int inputSize, int outputSize, double dropoutRate) : base(inputSize, outputSize)
    {
        DropoutRate = dropoutRate;
    }

    public override Tensor Forward(Tensor x)
    {
        CachedInput = x.Clone();

        if (!IsTraining)
        {
            return x;
        }

        int batchSize = x.Shape[0];

        Mask = new Tensor(batchSize, InputSize);
        Tensor output = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                Mask[i,j] = Utility.Random.NextDouble() > DropoutRate ? 1 : 0;

                output[i,j] = x[i,j] * Mask[i,j] / (1 - DropoutRate);
            }
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Backward(Tensor x)
    {
        int batchSize = x.Shape[0];

        Tensor gradInput = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                gradInput[i,j] = x[i,j] * Mask[i,j] / (1 - DropoutRate);
            }
        }

        return gradInput;
    }

    

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }
}