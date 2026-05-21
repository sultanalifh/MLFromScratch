using System.Text.Json.Serialization;

class Adam : Optimizer
{
    [JsonInclude]
    public List<Tensor> Velocities;
    [JsonInclude]
    public List<Tensor> Variances;

    [JsonInclude]
    public double Beta1;
    [JsonInclude]
    public double TBeta1;
    [JsonInclude]
    public double Beta2;
    [JsonInclude]
    public double TBeta2;
    
    public Adam(Sequential sequential, double learningRate, double beta1 = 0.9, double beta2 = 0.999) : base(sequential, learningRate)
    {
        Velocities = new List<Tensor>();
        Variances = new List<Tensor>();

        Beta1 = TBeta1 = beta1;
        Beta2 = TBeta2 = beta2;
       

        foreach(Parameter parameter in Parameters)
        {
            Tensor paramData = parameter.Data;

            int[] velocityShape = new int[paramData.Shape.Length];
            int[] varianceShape = new int[paramData.Shape.Length];

            Array.Copy(paramData.Shape, velocityShape, paramData.Shape.Length);
            Array.Copy(paramData.Shape, varianceShape, paramData.Shape.Length);

            Tensor velocity = new Tensor(velocityShape);
            Tensor variance = new Tensor(varianceShape);

            Velocities.Add(velocity);
            Variances.Add(variance);
        }
    }

    public override void Step()
    {
        int paramCount = Parameters.Count;

        for(int i = 0; i < paramCount; i++)
        {
            Parameter parameter = Parameters[i];

            Tensor velocity = Velocities[i];
            Tensor variance = Variances[i];

            double[] data = parameter.Data.Data;
            double[] grad = parameter.Grad.Data;

            double[] velocityData = velocity.Data;
            double[] varianceData = variance.Data;

            int gradSize = grad.Length;

            for(int j = 0; j < gradSize; j++)
            {
                velocityData[j] = velocityData[j] * Beta1 + grad[j] * (1 - Beta1);
                varianceData[j] = varianceData[j] * Beta2 + grad[j] * grad[j] * (1 - Beta2);

                double velHat = velocityData[j] / (1 - TBeta1);
                double varHat = varianceData[j] / (1 - TBeta2);

                data[j] -= LearningRate * velHat / (Math.Sqrt(varHat) + Utility.e5Eps);
            }
        }

        TBeta1 *= TBeta1;
        TBeta2 *= TBeta2;
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