class DenseLayer : Layer
{
    // Parameters
    public Matrix Weights;
    public Matrix Bias; 

    // Gradient cache
    public Matrix GradWeights;
    public Matrix GradBias;

    // Forward cache
    public Matrix CachedInput;
    public Matrix CachedOutput;

    public DenseLayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        Weights = new Matrix(outputSize, inputSize);
        Bias = new Matrix(outputSize, 1);

        GradWeights = new Matrix(outputSize, inputSize);
        GradBias = new Matrix(outputSize, 1);
    }

    public override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();
        x = x.Dot(Weights.Transpose());
        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                x[i,j] += Bias[j,0];
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

        Matrix backGrad = new Matrix(x.Rows, InputSize);

        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                double w_grad = x[i,j];
                for(int k = 0; k < InputSize; k++)
                {
                    GradWeights[j,k] += w_grad * CachedInput[i,k];
                    backGrad[i,k] += w_grad * Weights[j,k];
                }
                GradBias[j,0] += w_grad;
            }
        }

        return backGrad;
    }

    public override void LearnGradient(double learningRate)
    {
        for(int i = 0; i < OutputSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                Weights[i,j] -= GradWeights[i,j] * learningRate;
                GradWeights[i,j] = 0;
            }
            
            Bias[i,0] -= GradBias[i,0] * learningRate;
            GradBias[i,0] = 0;
        }
    }
}