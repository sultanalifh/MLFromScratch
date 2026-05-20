class LeakyReLU : ActivationLayer
{
    public LeakyReLU(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
    }

    public override Tensor Activate(Tensor x)
    {
        CachedInput = x.Clone();

        int batchSize = x.Shape[0];

        Tensor output = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                output[i,j] = Math.Max(x[i,j] * 0.01, x[i,j]);
            }
        }

        CachedOutput = x.Clone();

        return output;
    }

    public override Tensor Derivative(Tensor x)
    {
        int batchSize = x.Shape[0];

        Tensor gradInput = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                gradInput[i,j] = x[i,j] * (CachedInput[i,j] > 0 ? 1 : 0.01);
            }
        }

        return gradInput;
    }
}