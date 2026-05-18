using System.Text.Json.Serialization;

abstract class Optimizer
{
    public List<Parameter> Parameters;

    [JsonInclude]
    public double LearningRate;

    public Optimizer(Sequential sequential, double learningRate)
    {
        Parameters = [.. sequential.Parameters()];

        LearningRate = learningRate;
    }

    public abstract void Step();

    public abstract void ZeroGrad();
}