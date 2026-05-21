class SGD : Optimizer
{
    public SGD(Sequential sequential, double learningRate) : base(sequential, learningRate)
    {
    }

    public override void Step()
    {
        foreach(Parameter parameter in Parameters)
        {
            double[] data = parameter.Data.Data;
            double[] grad = parameter.Grad.Data;

            int dataSize = data.Length;

            for(int i = 0; i < dataSize; i++)
            {
                data[i] -= grad[i] * LearningRate;
            }
        }
    }

    public override void ZeroGrad()
    {
        foreach(Parameter parameter in Parameters)
        {
            Tensor grad = parameter.Grad;

            double[] data = grad.Data;
            int length = data.Length;

            for(int i = 0; i < length; i++)
            {
                data[i] = 0;
            }
        }
    }
}