class LeakyReLU : ActivationLayer
{
    public LeakyReLU(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        
    }

    public override Matrix Activate(Matrix x)
    {
        CachedInput = x.Clone();

        int batchSize = x.Rows;

        Matrix output = new Matrix(batchSize, OutputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                output[i,j] = Math.Max(0.01, x[i,j]);
            }
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Matrix Derivative(Matrix x)
    {
        int batchSize = x.Rows;
        
        Matrix gradInput = new Matrix(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                gradInput[i,j] = x[i,j] * (CachedInput[i,j] > 0 ? 1 : 0.01);
            }
        }

        return gradInput;
    }
}