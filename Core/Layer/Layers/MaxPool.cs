using System.Text.Json.Serialization;

class MaxPool : Layer
{
    [JsonInclude]
    public int PoolHeight;
    [JsonInclude]
    public int PoolWidth;
    [JsonInclude]
    public int Stride;
    [JsonInclude]
    public int Padding;
    public MaxPool(int inputSize, int outputSize, int poolHeight, int poolWidth, int stride, int padding) : base(inputSize, outputSize)
    {
        PoolHeight = poolHeight;
        PoolWidth = poolWidth;
        
        Stride = stride;
        Padding = padding;
    }

    public override Tensor Forward(Tensor x)
    {
        x = x.Pad(Padding);

        CachedInput = x.Clone();

        int[] shape = x.Shape;

        int batchSize = shape[0];
        int height = shape[2];
        int width = shape[3];

        int outputHeight = (height - PoolHeight) / Stride + 1;
        int outputWidth = (width - PoolWidth) / Stride + 1;

        Tensor output = new Tensor(batchSize, InputSize, outputHeight, outputWidth);

        Array.Fill(output.Data, int.MinValue);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                for(int k = 0; k < outputHeight; k++)
                {
                    for(int l = 0; l < outputWidth; l++)
                    {
                        for(int m = 0; m < PoolHeight; m++)
                        {
                            for(int n = 0; n < PoolWidth; n++)
                            {
                                int x_pos = l * Stride + n;
                                int y_pos = k * Stride + m;

                                output[i,j,k,l] = Math.Max(output[i,j,k,l], x[i,j,y_pos,x_pos]);
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
        int[] inputShape = CachedInput.Shape;
        int[] shape = CachedOutput.Shape;

        int batchSize = shape[0];

        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        int windowHeight = shape[2];
        int windowWidth = shape[3];

        Tensor gradInput = new Tensor(batchSize, InputSize, inputHeight, inputWidth);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < InputSize; j++)
            {
                for(int k = 0; k < windowHeight; k++)
                {
                    for(int l = 0; l < windowWidth; l++)
                    {
                        int reward = 1;

                        for(int m = 0; m < PoolHeight; m++)
                        {
                            for(int n = 0; n < PoolWidth; n++)
                            {
                                int x_pos = l * Stride + n;
                                int y_pos = k * Stride + m;

                                gradInput[i,j,y_pos,x_pos] += x[i,j,k,l] * (CachedInput[i,j,y_pos,x_pos] == CachedOutput[i,j,k,l] ? reward : 0);

                                if(CachedInput[i,j,y_pos,x_pos] == CachedOutput[i, j, k, l])
                                {
                                    reward = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        gradInput = gradInput.Pad(-Padding);

        return gradInput;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }
}