// Todo: Conv2D

using System.Text.Json.Serialization;

class Conv2D : Layer
{
    [JsonInclude]
    public Parameter Filter;
    [JsonInclude]
    public int FilterWidth;
    [JsonInclude]
    public int FilterHeight;
    [JsonInclude]
    public int Stride;
    [JsonInclude]
    public int Padding;

    public Conv2D(int inputSize, int outputSize, int filterWidth, int filterHeight, int stride = 1, int padding = 0) : base(inputSize, outputSize)
    {
        Filter = new Parameter("Filter", outputSize, inputSize, filterHeight, filterWidth);

        FilterWidth = filterWidth;
        FilterHeight = filterHeight;

        Stride = stride;
        Padding = padding;
    }
    public override Tensor Forward(Tensor x)
    {
        CachedInput = x.Clone();

        int[] shape = x.Shape;

        int batchSize = shape[0];
        int height = shape[2];
        int width = shape[3];

        int outputHeight = (height - FilterHeight) / Stride + 1;
        int outputWidth = (width - FilterWidth) / Stride + 1;

        Tensor output = new Tensor(batchSize, OutputSize, outputHeight, outputWidth);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                for(int k = 0; k < InputSize; k++)
                {
                    for(int l = 0; l < outputHeight; l++)
                    {
                        for(int m = 0; m < outputWidth; m++)
                        {
                            for(int n = 0; n < FilterHeight; n++)
                            {
                                for(int o = 0; o < FilterWidth; o++)
                                {
                                    int x_pos = m * Stride + o;
                                    int y_pos = l * Stride + n;

                                    output[i,j,l,m] += x[i,k,y_pos,x_pos] * Filter.Data[j,k,n,o];
                                }
                            }
                        }
                    }
                }
            }
        }

        CachedOutput = output.Clone();

        return output;
    }
    public override Tensor Backward(Tensor x)
    {
        int[] shape = CachedOutput.Shape;
        int[] inputShape = CachedInput.Shape;

        int batchSize = shape[0];

        int outputHeight = shape[2];
        int outputWidth = shape[3];
        
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        Tensor gradInput = new Tensor(batchSize, InputSize, inputHeight, inputWidth);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < OutputSize; j++)
            {
                for(int k = 0; k < InputSize; k++)
                {
                    for(int l = 0; l < outputHeight; l++)
                    {
                        for(int m = 0; m < outputWidth; m++)
                        {
                            for(int n = 0; n < FilterHeight; n++)
                            {
                                for(int o = 0; o < FilterWidth; o++)
                                {
                                    int x_pos = m * Stride + o;
                                    int y_pos = l * Stride + n;

                                    gradInput[i,k,y_pos,x_pos] += x[i,j,l,m] * Filter.Data[j,k,n,o];
                                    Filter.Grad[j,k,n,o] += x[i,j,l,m] * CachedInput[i,k,y_pos,x_pos];
                                }
                            }
                        }
                    }
                }
            }
        }

        return gradInput;
    }
    public override IEnumerable<Parameter> Parameters()
    {
        yield return Filter;
    }
}