class Softmax : ActivationLayer
{
    public Softmax(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
    }

    public override Matrix Activate(Matrix x)
    {
        CachedInput = x.Clone();

        int BatchSize = x.Rows;

        Matrix output = new Matrix(BatchSize, OutputSize);
        
        for(int i = 0; i < BatchSize; i++)
        {
            double max = Utility.e12Eps;

            for(int j = 0; j < InputSize; j++)
            {
                max = Math.Max(max, x[i,j]);
            }

            double sum = 0;

            for(int j = 0; j < InputSize; j++)
            {
                double zj = Math.Exp(x[i,j] - max);
                sum += zj;

                output[i,j] = zj;
            }

            for(int j = 0; j < InputSize; j++)
            {
                double pj = output[i,j] / sum;
                output[i,j] = pj;
            }
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Matrix Derivative(Matrix x)
    {
        int BatchSize = x.Rows;

        Matrix gradInput = new Matrix(BatchSize, InputSize);

        for(int i = 0; i < BatchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                for(int k = 0; k < OutputSize; k++)
                {
                    if(j == k) gradInput[i,j] += x[i,k] * CachedOutput[i,j] * (1 - CachedOutput[i,k]);
                    else gradInput[i,j] += x[i,k] * -CachedOutput[i,j] * CachedOutput[i,k];
                }
            }
        }
        
        return gradInput;
    }
}