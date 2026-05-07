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

    protected override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();
        x = x.Dot(Weights.Transpose());
        for(int i = 0; i < Bias.Rows; i++)
        {
            for(int j = 0; j < Bias.Cols; j++)
            {
                x[i,j] += Bias[i,j];
            }
        }
        CachedOutput = x.Clone();
        return x;
    }

    protected override Matrix Backward(Matrix x)
    {
        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < Weights.Rows; j++)
            {
                for(int k = 0; k < Weights.Cols)
                {
                    GradWeights[j,k] += x[i,0] * CachedInput[]
                }
            }
        }
        return default;
    }
}