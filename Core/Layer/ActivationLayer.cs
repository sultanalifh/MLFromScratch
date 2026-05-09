abstract class ActivationLayer : Layer
{
    public Matrix CachedInput;
    public Matrix CachedOutput;
    public ActivationLayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }

    public override Matrix Backward(Matrix x) => Derivative(x);

    public override Matrix Forward(Matrix x) => Activate(x);

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }

    public override void Step(double learningRate)
    {
        return;
    }

    public abstract Matrix Activate(Matrix x);

    public abstract Matrix Derivative(Matrix x);
}