abstract class Layer
{
    public int InputSize;
    public int OutputSize;

    public Matrix CachedInput;
    public Matrix CachedOutput;

    public const double Lambda = 0.001;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }
    public abstract Matrix Forward(Matrix x);

    public abstract Matrix Backward(Matrix x);

    public abstract void Step(double learningRate);

    public abstract IEnumerable<Parameter> Parameters();
}