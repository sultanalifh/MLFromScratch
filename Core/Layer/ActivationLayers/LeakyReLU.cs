class LeakyReLU : ActivationLayer
{
    public LeakyReLU() : base(0,0)
    {
    }

    public override Tensor Activate(Tensor x)
    {
        CachedInput = x.Clone();

        int dataSize = x.Data.Length;

        Tensor output = x.Clone();

        for(int i = 0; i < dataSize; i++)
        {
            output.Data[i] = Math.Max(0.01 * x.Data[i], x.Data[i]);
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Derivative(Tensor x)
    {
        int dataSize = x.Data.Length;

        Tensor gradInput = x.Clone();

        for(int i = 0; i < dataSize; i++)
        {
            gradInput.Data[i] = x.Data[i] * (CachedInput.Data[i] > 0 ? 1 : 0.01);
        }

        return gradInput;
    }
}