
class SigmoidLayer : Layer
{
    public Matrix CachedOutput;
    public SigmoidLayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }

    public override Matrix Forward(Matrix x)
    {
        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < x.Cols; j++)
            {
                x[i,j] = 1 / (1 + Math.Exp(-x[i,j]));
            }
        }
        
        CachedOutput = x.Clone();

        return x;
    }

    public override Matrix Backward(Matrix x)
    {
        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < x.Cols; j++)
            {
                x[i,j] *= CachedOutput[i,j] * (1 - CachedOutput[i,j]);
            }
        }

        return x;
    }

    public override void Step(double learningRate)
    {
        return;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }
}