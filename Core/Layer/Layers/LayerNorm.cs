using System.Text.Json.Serialization;

class LayerNorm : Layer
{
    [JsonInclude]
    public Parameter Gamma;
    
    [JsonInclude]
    public Parameter Beta;


    public LayerNorm(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        Gamma = new Parameter("Gamma", 1, inputSize);
        Beta = new Parameter("Beta", 1, inputSize);

        Array.Fill(Gamma.Data.Data, 1);
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
                double x_norm = (x[i,j] - mean) / std;
                double x_hat = Gamma.Data[0,j] * x_norm + Beta.Data[0,j];

                output[i,j] = x_hat;
            }
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Backward(Tensor x)
    {
        // Gamma / std

        // E(Gamma * -1/std) * 1/n
        // E(Gamma) / n * -1/std
        // Mean(Gamma) * -1 / std


        // E(Gamma * -(xi - u)/std^2) * 1/2std * 2(xi - u)/n
        // E(Gamma * x_hat_j) * -1/std * 1/2std * 2(xi - u)/n
        // Mean(Gamma * x_hat_j) * -1/std * 1/2std * 2(xi - u)
        // Mean(Gamma * x_hat_j) * -1/std * x_hat_i

        //  Gamma / std - Mean(Gamma) / std - Mean(gamma * x_hat j) * x_hat_i / std
        //  (Gamma - Mean(Gamma) - Mean(Gamma * x_hat_j) * x_hat_i) / std

        int batchSize = x.Shape[0];

        Tensor gradInput = new Tensor(batchSize, InputSize);

        for(int i = 0; i < batchSize; i++)
        {
            double mean = 0;

            for(int j = 0; j < InputSize; j++)
            {
                mean += CachedInput[i,j];
            }

            mean /= InputSize;

            double variances = 0;

            for(int j = 0; j < InputSize; j++)
            {
                variances += (CachedInput[i,j] - mean) * (CachedInput[i,j] - mean);
            }

            variances /= InputSize;

            double std = Math.Sqrt(variances + Utility.e5Eps);

            double[] x_norms = new double[InputSize];
            double xHat = 0;
            double xHatxHat = 0;

            for(int j = 0; j < InputSize; j++)
            {
                double x_norm = (CachedInput[i,j] - mean) / std;

                x_norms[j] = x_norm;

                Gamma.Grad[0,j] += x[i,j] * x_norm;
                Beta.Grad[0,j] += x[i,j];

                xHat += x[i,j] * Gamma.Data[0,j];
                xHatxHat += x[i,j] * Gamma.Data[0,j] * x_norm;
            }

            xHat /= InputSize;
            xHatxHat /= InputSize;

            for(int j = 0; j < InputSize; j++)
            {
                double x_hat_j = x[i,j] * Gamma.Data[0,j];

                gradInput[i,j] = (x_hat_j - xHat - xHatxHat * x_norms[j]) / std;
            }
        }

        return gradInput;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield return Gamma;
        yield return Beta;
    }
}