// mean
// variances
// meanabs
// maxabs
// hasnan
// zeroratio

static class Stats
{
    public static double Mean(Tensor tensor)
    {
        double sum = 0;

        double[] data = tensor.Data;
        int dataLength = data.Length;

        for(int i = 0; i < dataLength; i++)
        {
            sum += data[i];
        }

        sum /= dataLength;

        return sum;
    }

    public static double Variance(Tensor tensor)
    {
        double mean = Mean(tensor);
        double variances = 0;

        double[] data = tensor.Data;
        int dataLength = data.Length;

        for(int i = 0; i < dataLength; i++)
        {
            variances += (data[i] - mean) * (data[i] - mean);
        }
        
        variances /= dataLength;

        return variances;
    }

    public static double MeanAbs(Tensor tensor)
    {
        double sum = 0;

        double[] data = tensor.Data;
        int dataLength = data.Length;


        for(int i = 0; i < dataLength; i++)
        {
            sum += Math.Abs(data[i]);
        }

        sum /= dataLength;

        return sum;
    }

    public static double MaxAbs(Tensor tensor)
    {
        double max = 0;

        double[] data = tensor.Data;
        int dataLength = data.Length;

        for(int i = 0; i < dataLength; i++)
        {
            max = Math.Max(max, Math.Abs(data[i]));
        }

        return max;
    }

    public static bool HasNaN(Tensor tensor)
    {
        double[] data = tensor.Data;
        int dataLength = data.Length;

        for(int i = 0; i < dataLength; i++)
        {
            if(data[i] == double.NaN)
            {
                return true;
            }
        }

        return false;
    }

    public static double ZeroRatio(Tensor tensor)
    {
        double[] data = tensor.Data;
        int dataLength = data.Length;

        double zero = 0;

        for(int i = 0; i < dataLength; i++)
        {
            if(data[i] == zero)
            {
                zero++;
            }
        }

        return zero / dataLength;
    }
}