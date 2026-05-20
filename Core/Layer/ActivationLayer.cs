abstract class ActivationLayer : Layer
{
    public ActivationLayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        
    }

    public override Tensor Backward(Tensor x) => Derivative(x);

    public override Tensor Forward(Tensor x) => Activate(x);

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }
    
    public abstract Tensor Activate(Tensor x);

    public abstract Tensor Derivative(Tensor x);
}