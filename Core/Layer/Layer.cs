using System.Text.Json.Serialization;

abstract class Layer
{
    [JsonInclude]
    public int InputSize;

    [JsonInclude]
    public int OutputSize;

    public Tensor CachedInput;

    public Tensor CachedOutput;

    public double Lambda = 0;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }
    public abstract Tensor Forward(Tensor x);

    public abstract Tensor Backward(Tensor x);

    public abstract IEnumerable<Parameter> Parameters();
}