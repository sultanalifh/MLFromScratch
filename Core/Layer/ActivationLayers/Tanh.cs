class Tanh : ActivationLayer
{
    public Tanh(int inputSize, int outputSize) : base(inputSize, outputSize)
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
                double pz = Math.Exp(x[i,j]);
                double nz = Math.Exp(-x[i,j]);
                
                output[i,j] = (pz - nz) / (pz + nz);
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
                gradInput[i,j] = 1 - CachedOutput[i,j] * CachedOutput[i,j];
            }
        }

        return gradInput;
    }
}