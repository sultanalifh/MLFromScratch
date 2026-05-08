
class ReLULayer : Layer 
{
    public Matrix CachedInput;

    public ReLULayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;

        CachedInput = new Matrix(outputSize, inputSize);
    }
    public override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();
        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < x.Cols; j++)
            {
                x[i,j] = Math.Max(0, x[i,j]);
            }
        }
        return x;
    }

    public override Matrix Backward(Matrix x)
    {
        Matrix backGrad = new Matrix(x.Rows, x.Cols);

        for(int i = 0; i < x.Rows; i++)
        {
            for(int j = 0; j < x.Cols; j++)
            {
                backGrad[i,j] = CachedInput[i,j] > 0 ? x[i,j] : 0;
            }
        }

        return backGrad;
    }

    public override void LearnGradient(double learningRate)
    {
        return;
    }
}