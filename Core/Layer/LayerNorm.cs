
using System.Diagnostics.CodeAnalysis;

class LayerNorm : Layer
{
    private double _Epsilon = 1e-5;
    public double Gamma = 1;
    public double GammaGrad = 0;
    public double Beta = 0;
    public double BetaGrad = 0;
    public Matrix CachedInput;
    public Matrix CachedOutput;

    public LayerNorm(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        CachedInput = new Matrix(outputSize, inputSize);
        CachedOutput = new Matrix(outputSize, inputSize);
    }
    public override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();

        int BatchSize = x.Rows;

        for(int i = 0; i < BatchSize; i++)
        {
            double Mean = x.RowSum(i) / InputSize;

            double Variance = 0;

            for(int j = 0; j < InputSize; j++)
            {
                Variance += (x[i,j] - Mean) * (x[i,j] - Mean);
            }

            Variance /= InputSize;

            double std = Math.Sqrt(Variance + _Epsilon);

            for(int j = 0; j < InputSize; j++)
            {
                double x_norm = (x[i,j] - Mean) / std;

                double x_normalized = Gamma * x_norm + Beta;

                x[i,j] = x_normalized;
            }
        }

        CachedOutput = x.Clone();

        return x;
    }
    public override Matrix Backward(Matrix x)
    {

        // dyi/dxi from xnorm: Gamma / std
        // dyi/dxi from mean: dyi/dxnorm * dxnorm/du * du/dxi
        // = Sum(dyi/dxnorm) * -1/std * 1/n
        // = -mean(dyi/dxnorm)/std
        // dyi/dxi from variance: dyi/dxnorm * dxnorm/dstd * dstd/do^2 * do^2/dxi
        // = -1/std * Sum(dyi * Gamma * xhat) * 1/2std * 2(xi - u)/n
        // = -1/std * mean(dyi * Gamma * xhat) * xhat
        // = -xhat * mean(dyi * Gamma * xhat) / std
        
        // total = Gamma * 1/std - mean(dyi * Gamma)/std - xhat_i * mean(dyi * Gamma * xhat OR dxHatxhat)/std

        int BatchSize = x.Rows;

        for(int i = 0; i < BatchSize; i++)
        {
            double[] xhat = new double[InputSize];
            double[] gradXhat = new double[InputSize];

            double dXhatXhat = 0;
            double dXhat = 0;

            double mean = CachedInput.RowSum(i) / InputSize;

            double variance = 0;

            for(int j = 0; j < InputSize; j++)
                variance += (CachedInput[i,j] - mean) * (CachedInput[i,j] - mean);

            variance /= InputSize;

            double std = Math.Sqrt(variance + _Epsilon);

            for(int j = 0; j < InputSize; j++)
            {
                double x_norm = (CachedInput[i,j] - mean) / std;
                double dyi_dxnorm = x[i,j] * Gamma;
                gradXhat[j] = dyi_dxnorm;
                xhat[j] = x_norm;
                dXhat += dyi_dxnorm;
                dXhatXhat += dyi_dxnorm * x_norm;

                GammaGrad += x[i,j] * x_norm;
                BetaGrad += x[i,j];
            }

            dXhat /= InputSize;
            dXhatXhat /= InputSize;

            for(int j = 0; j < InputSize; j++)
            {
                x[i,j] = (gradXhat[j] - dXhat - xhat[j] * dXhatXhat) / std;
            }
        }

        return x;
    }
    public override void LearnGradient(double learningRate)
    {
        Gamma -= GammaGrad * learningRate;
        Beta -= BetaGrad * learningRate;

        GammaGrad = BetaGrad = 0;
    }
}