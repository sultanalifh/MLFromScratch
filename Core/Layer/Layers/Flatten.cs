using System.Text.Json.Serialization;

class Flatten : Layer
{
    public Flatten(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
    }

    public override Tensor Forward(Tensor x)
    {
        CachedInput = x.Clone();

        int[] shape = x.Shape;

        int batchSize = shape[0];
        int flattenSize = 1;

        for(int i = 1; i < shape.Length; i++)
        {
            flattenSize *= shape[i];
        }

        Tensor output = x.Reshape(batchSize, flattenSize);

        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Backward(Tensor x)
    {
        int[] inputShape = CachedInput.Shape;
        
        Tensor gradInput = x.Reshape(inputShape);

        return gradInput;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }
}