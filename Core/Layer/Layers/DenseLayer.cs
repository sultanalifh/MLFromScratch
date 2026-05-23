using System.Text.Json.Serialization;

class DenseLayer : Layer
{
    [JsonInclude]
    public Parameter Weights;

    [JsonInclude]
    public Parameter Bias;

    public DenseLayer(int inputSize, int outputSize) : base(inputSize, outputSize)
    {
        Weights = new Parameter("Weight", new Matrix(outputSize, inputSize));

        Bias = new Parameter("Bias", new Matrix(outputSize, 1));
    }

    public override Tensor Forward(Tensor x)
    {
        int[] shape = x.Shape;

        int inputDimension = shape.Length;
        int batchSize = shape[0];

        if(inputDimension != 2)
        {
            throw new ArgumentException("Cannot forward non-matrix like input!");
        }

        CachedInput = x.Clone();

        Tensor output = x.Dot(Weights.Data.Transpose());

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                output[i,j] += Bias.Data[j,0];
            }
        }

        CachedOutput = output.Clone();

        return output;
    }

    public override Tensor Backward(Tensor x)
    {
        int batchSize = x.Shape[0];

        Tensor inputGrad = new Tensor(batchSize, InputSize);
    
        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                for(int k = 0; k < InputSize; k++)
                {
                    inputGrad[i,k] += x[i,j] * Weights.Data[j,k];
                    Weights.Grad[j,k] += x[i,j] * CachedInput[i,k] + 2 * Lambda * Weights.Data[j,k];
                }
                Bias.Grad[j,0] += x[i,j];
            }
        }

        return inputGrad;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield return Weights;
        yield return Bias;
    }
}