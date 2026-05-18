using System.Text.Json.Serialization;

abstract class Layer
{
    [JsonInclude]
    public int InputSize;

    [JsonInclude]
    public int OutputSize;

    public Matrix CachedInput;

    public Matrix CachedOutput;

    public double Lambda = 0;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }
    public abstract Matrix Forward(Matrix x);

    public abstract Matrix Backward(Matrix x);

    public abstract void Step(double learningRate);

    public abstract IEnumerable<Parameter> Parameters();
}