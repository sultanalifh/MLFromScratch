static class Stats
{
    public static double Mean(Matrix x)
    {
        int Rows = x.Rows;
        int cols = x.Cols;

        Matrix mean = new Matrix(Rows, 1);

        for(int i = 0; i < Rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                mean[i,0] += x[i,j];
            }

            mean[i,0] /= cols;
        }

        double meanSum = 0;

        for(int i = 0; i < Rows; i++)
        {
            meanSum += mean[i,0];
        }

        meanSum /= Rows;

        return meanSum;
    }

    public static double Variance(Matrix x)
    {
        int rows = x.Rows;
        int cols = x.Cols;

        Matrix mean = new Matrix(rows, 1);

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                mean[i,0] += x[i,j];
            }

            mean[i,0] /= cols;
        }

        Matrix variance = new Matrix(rows,1);

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                variance[i,0] += (x[i,j] - mean[i,0]) * (x[i,j] - mean[i,0]);
            }

            variance[i,0] = Math.Sqrt(variance[i,0] / cols + Utility.e5Eps);
        }

        double varSum = 0;

        for(int i = 0; i < rows; i++)
        {
            varSum += variance[i,0];
        }

        varSum /= rows;

        return varSum;
    }

    
    public static double MeanAbs(Matrix x)
    {
        int rows = x.Rows;
        int cols = x.Cols;

        Matrix meanAbs = new Matrix(rows, 1);

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                meanAbs[i,0] += Math.Abs(x[i,j]);
            }

            meanAbs[i,0] /= cols;
        }

        double meanAbsSum = 0;

        for(int i = 0; i < rows; i++)
        {
            meanAbsSum += meanAbs[i,0];
        }

        meanAbsSum /= rows;

        return meanAbsSum;
    }

    public static double MaxAbs(Matrix x)
    {
        int rows = x.Rows;
        int cols = x.Cols;

        double max = 0;

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                max = Math.Max(max, Math.Abs(x[i,j]));
            }
        }

        return max;
    }

    public static bool HasNaN(Matrix x)
    {
        int rows = x.Rows;
        int cols = x.Cols;

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                if(x[i,j] == double.NaN)
                {
                    return true;
                }
            }
        }

        return false;
    }

    public static double ZeroRatio(Matrix x)
    {
        int rows = x.Rows;
        int cols = x.Cols;

        int totalElem = rows * cols;
        double totalZero = 0;

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                if(x[i,j] == 0) totalZero++;
            }
        }

        return totalZero / totalElem;
    }
}