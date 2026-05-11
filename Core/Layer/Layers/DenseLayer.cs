class DenseLayer : Layer
{
    public Parameter Weights;
    public Parameter Bias; 
    public DenseLayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        Weights = new Parameter(outputSize, inputSize, "Weight");
        Bias = new Parameter(outputSize, 1, "Bias");
    }

    public override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();
        x = x.Dot(Weights.Data.Transpose());
        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                x[i,j] += Bias.Data[j,0];
            }
        }
        CachedOutput = x.Clone();
        return x;
    }

    public override Matrix Backward(Matrix x)
    {
        if(x.Rows != CachedOutput.Rows)
        {
            throw new ArgumentException("Backward gradient output size was not the same!");
        }

        Matrix gradInput = new Matrix(x.Rows, InputSize);

        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                double w_grad = x[i,j];
                for(int k = 0; k < InputSize; k++)
                {
                    Weights.Grad[j,k] += w_grad * CachedInput[i,k];
                    gradInput[i,k] += w_grad * Weights.Data[j,k];
                }
                Bias.Grad[j,0] += w_grad;
            }
        }
        
        return gradInput;
    }

    public override void Step(double learningRate)
    {
        for(int i = 0; i < OutputSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                Weights.Data[i,j] -= Weights.Grad[i,j] * learningRate;
                Weights.Grad[i,j] = 0;
            }
            
            Bias.Data[i,0] -= Bias.Grad[i,0] * learningRate;
            Bias.Grad[i,0] = 0;
        }
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield return Weights;
        yield return Bias;
    }
}