class Sigmoid : ActivationLayer
{
    public Sigmoid(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
    }

    public override Tensor Activate(Tensor x)
    {
        int batchSize = x.Shape[0];

        CachedInput = x.Clone();

        Tensor output = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                output[i,j] = 1 / (1 + Math.Exp(-x[i,j]));
            }
        }

        CachedOutput = output.Clone();

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
                gradInput[i,j] = x[i,j] * CachedOutput[i,j] * (1 - CachedOutput[i,j]);
            }
        }
        
        return gradInput;
    }
}