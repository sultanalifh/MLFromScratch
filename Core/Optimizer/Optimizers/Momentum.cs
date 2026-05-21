using System.Text.Json.Serialization;

class Momentum : Optimizer
{
    [JsonInclude]
    public List<Tensor> Velocities;

    [JsonInclude]
    public double momentum;
    public Momentum(Sequential sequential, double learningRate, double momentum = 0.9) : base(sequential, learningRate)
    {
        Velocities = new List<Tensor>();

        this.momentum = momentum;

        foreach(Parameter parameter in Parameters)
        {
            int[] shape = parameter.Data.Shape;
            int[] clonedShape;

            int shapeSize = shape.Length;

            clonedShape = new int[shapeSize];

            Array.Copy(shape, clonedShape, shapeSize);

            Tensor velocity = new Tensor(clonedShape);

            Velocities.Add(velocity);
        }
    }

    public override void Step()
    {
        int paramCount = Parameters.Count;

        for(int i = 0; i < paramCount; i++)
        {
            Parameter parameter = Parameters[i];
            Tensor velocity = Velocities[i];

            double[] data = parameter.Data.Data;
            double[] grad = parameter.Grad.Data;

            double[] velocityData = velocity.Data;

            int dataSize = data.Length;

            for(int j = 0; j < dataSize; j++)
            {
                velocityData[j] = momentum * velocityData[j] + grad[j] * LearningRate;

                data[j] -= velocityData[j];
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