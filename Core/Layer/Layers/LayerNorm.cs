class LayerNorm : Layer
{
    public Parameter Gamma;
    public Parameter Beta;


    public LayerNorm(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        Gamma = new Parameter("Gamma", outputSize, 1);
        Beta = new Parameter("Beta", outputSize, 1);

    }
    public override Tensor Forward(Tensor x)
    {
        CachedInput = x.Clone();

        int batchSize = x.Shape[0];

        Tensor output = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            double mean = 0;

            for(int j = 0; j < InputSize; j++)
            {
                mean += x[i,j];
            }

            mean /= InputSize;

            double variances = 0;

            for(int j = 0; j < InputSize; j++)
            {
                variances += (x[i,j] - mean) * (x[i,j] - mean);
            }


            // 1/2 * x^-1/2
            // 1/2std
            variances /= InputSize;

            double std = Math.Sqrt(variances + Utility.e5Eps);

            for(int j = 0; j < InputSize; j++)
            {
                double x_hat = (x[i,j] - mean) / std;
                double x_norm = Gamma.Data[i,0] * x_hat + Beta.Data[i,0];

                output[i,j] = x_norm;
            }
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Backward(Tensor x)
    {
        
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield return Gamma;
        yield return Beta;
    }
}