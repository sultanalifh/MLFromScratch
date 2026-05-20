class Softmax : ActivationLayer
{
    public Softmax(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
    }

    public override Tensor Activate(Tensor x)
    {
        CachedInput = x.Clone();

        int batchSize = x.Shape[0];

        Tensor output = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            double max = x[i,0];

            for(int j = 0; j < InputSize; j++)
            {
                max = Math.Max(max, x[i,j]);
            }

            double sum = 0;

            for(int j = 0; j < InputSize; j++)
            {
                output[i,j] = Math.Exp(x[i,j] - max);

                sum += output[i,j];
            }

            for(int j = 0; j < InputSize; j++)
            {
                output[i,j] = output[i,j] / sum;
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
                for(int k = 0; k < InputSize; k++)
                {
                    if(j == k)
                    {
                        gradInput[i,j] += CachedOutput[i,j] * (1 - CachedOutput[i,j]);
                    }
                    else
                    {
                        gradInput[i,j] += CachedOutput[i,j] * CachedOutput[i,k];
                    }
                }
            }
        }

        return gradInput;
    }
}