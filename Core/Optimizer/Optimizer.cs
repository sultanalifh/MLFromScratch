abstract class Optimizer
{
    public List<Parameter> Parameters;

    public double LearningRate;

    public Optimizer(Sequential sequential, double learningRate)
    {
        Parameters = new List<Parameter>();

        foreach(Layer layer in sequential.Layers)
        {
            foreach(Parameter parameter in layer.Parameters())
            {
                Parameters.Add(parameter);
            }
        }

        LearningRate = learningRate;
    }

    public abstract void Step();

    public abstract void ZeroGrad();
}