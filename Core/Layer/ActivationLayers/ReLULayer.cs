
class ReLULayer : ActivationLayer
{
    public ReLULayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }

    public override Matrix Activate(Matrix x)
    {
        CachedInput = x.Clone();

        int BatchSize = x.Rows;

        Matrix output = new Matrix(BatchSize, OutputSize);

        for(int i = 0; i < BatchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                output[i,j] = Math.Max(0, x[i,j]);
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
            for(int j = 0; j < OutputSize; j++)
            {
                gradInput[i,j] = x[i,j] * (CachedInput[i,j] > 0 ? 1 : 0);
            }
        }

        return gradInput;
    }
}