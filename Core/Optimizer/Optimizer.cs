abstract class Optimizer
{
    public List<Parameter> Parameters;

    public double LearningRate;

    public Optimizer(Sequential sequential, double learningRate)
    {
        Parameters = new List<Parameter>();

        foreach(Parameter param in sequential.Parameters())
        {
            Parameters.Add(param);
        }

        LearningRate = learningRate;
    }

    public abstract void Step();

    public abstract void ZeroGrad();
}