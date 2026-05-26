using System.Text.Json.Serialization;

class Dropout : Layer
{
    [JsonInclude]
    public double DropoutRate;

    public Tensor Mask;
    public bool IsTraining;


    public Dropout(int inputSize, int outputSize, double dropoutRate) : base(inputSize, outputSize)
    {
        DropoutRate = dropoutRate;
    }

    public override Tensor Forward(Tensor x)
    {
        CachedInput = x.Clone();

        if (!IsTraining)
        {
            return x;
        }

        int[] inputShape = x.Shape;
        int[] shape = new int[inputShape.Length];
        double[] inputData = x.Data;

        Array.Copy(inputShape, shape, inputShape.Length);

        Tensor output = new Tensor(shape);
        Mask = output.Clone();

        for(int i = 0; i < inputData.Length; i++)
        {
            Mask.Data[i] = Utility.Random.NextDouble() > DropoutRate ? 1 : 0;

            output.Data[i] = inputData[i] * Mask.Data[i] / (1 - DropoutRate);
        }


        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Backward(Tensor x)
    {
        if (!IsTraining)
        {
            return x;
        }

        double[] data = x.Data;

        Tensor gradInput = x.Clone();

        for(int i = 0; i < data.Length; i++)
        {
            gradInput.Data[i] = data[i] * Mask.Data[i] / (1 - DropoutRate);
        }

        return gradInput;
    }

    

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }
}