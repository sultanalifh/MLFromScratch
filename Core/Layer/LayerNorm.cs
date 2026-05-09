
using System.Diagnostics.CodeAnalysis;

class LayerNorm : Layer
{
    private double _Epsilon = 1e-5;
    public Parameter Gamma;
    public Parameter Beta;
    public Matrix CachedInput;
    public Matrix CachedOutput;

    public LayerNorm(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;

        Gamma = new Parameter(1, inputSize);
        Beta = new Parameter(1, inputSize);

        for(int i = 0; i < InputSize; i++)
        {
            Gamma.Data[0,i] = 1;
        }

        CachedInput = new Matrix(outputSize, inputSize);
        CachedOutput = new Matrix(outputSize, inputSize);
    }
    public override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();

        int BatchSize = x.Rows;

        Matrix output = new Matrix(BatchSize, OutputSize);

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

                double x_normalized = Gamma.Data[0,j] * x_norm + Beta.Data[0,j];

                output[i,j] = x_normalized;
            }
        }

        CachedOutput = output.Clone();

        return output;
    }
    public override Matrix Backward(Matrix x)
    {
        int BatchSize = x.Rows;

        Matrix gradInput = new Matrix(BatchSize, InputSize);

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
                double dyi_dxnorm = x[i,j] * Gamma.Data[0,j];
                gradXhat[j] = dyi_dxnorm;
                xhat[j] = x_norm;
                dXhat += dyi_dxnorm;
                dXhatXhat += dyi_dxnorm * x_norm;

                Gamma.Grad[0,j] += x[i,j] * x_norm;
                Beta.Grad[0,j] += x[i,j];
            }

            dXhat /= InputSize;
            dXhatXhat /= InputSize;

            for(int j = 0; j < InputSize; j++)
            {
                gradInput[i,j] = (gradXhat[j] - dXhat - xhat[j] * dXhatXhat) / std;
            }
        }

        return gradInput;
    }
    public override void Step(double learningRate)
    {
        for(int i = 0; i < InputSize; i++)
        {
            Gamma.Data[0,i] -= Gamma.Grad[0,i] * learningRate;
            Gamma.Data[0,i] -= Beta.Grad[0,i] * learningRate;

            Gamma.Grad[0,i] = Beta.Grad[0,i] = 0;
        }
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield return Gamma;
        yield return Beta;
    }
}