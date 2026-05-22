class Tanh : ActivationLayer
{
    public Tanh() : base(0, 0)
    {
    }

    public override Tensor Activate(Tensor x)
    {
        CachedInput = x.Clone();

        int dataSize = x.Data.Length;

        Tensor output = x.Clone();

        for(int i = 0; i < dataSize; i++)
        {
            double pz = Math.Exp(x.Data[i]);
            double nz = Math.Exp(-x.Data[i]);

            output.Data[i] = (pz - nz) / (pz + nz);
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
            gradInput.Data[i] = 1 - CachedOutput.Data[i] * CachedOutput.Data[i];
        }

        return gradInput;
    }
}