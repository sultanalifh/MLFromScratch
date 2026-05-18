using System.Text.Json.Serialization;

class Dropout : Layer
{
    [JsonInclude]
    public double DropoutRate;
    public Matrix Mask;
    public bool IsTraining;

    public Dropout(int inputSize, int outputSize, double dropOutRate) : base(inputSize, outputSize)
    {
        DropoutRate = dropOutRate;

        IsTraining = true;
    }

    public override Matrix Forward(Matrix x)
    {
        CachedInput = x.Clone();

        if (!IsTraining)
        {
            CachedOutput = x.Clone();
            
            return CachedOutput;
        }

        int batchSize = x.Rows;

        Matrix output = new Matrix(batchSize, OutputSize);

        Mask = new Matrix(batchSize, OutputSize);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                Mask[i,j] = Utility.Random.NextDouble() > DropoutRate ? 1 : 0;

                output[i,j] = x[i,j] * Mask[i,j] / (1 - DropoutRate);
            }
        }

        CachedOutput = output.Clone();

        return CachedOutput;
    }

    public override Matrix Backward(Matrix x)
    {
        int BatchSize = x.Rows;

        Matrix gradInput = new Matrix(BatchSize, InputSize);

        if (!IsTraining)
        {
            gradInput = x.Clone();

            return gradInput;
        }

        for(int i = 0; i < BatchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                gradInput[i,j] = x[i,j] * Mask[i,j] / (1 - DropoutRate);
            }
        }

        return gradInput;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }

    public override void Step(double learningRate)
    {
        return;
    }
}