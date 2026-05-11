static class Loss
{
    public static Matrix BinaryCrossEntropyGrad(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows)
        {
            throw new ArgumentException("Number of predictions was not the same!");
        }

        int BatchSize = yPred.Rows;

        Matrix Result = new Matrix(BatchSize, 1);

        for(int i = 0; i < BatchSize; i++)
        {
            Result[i,0] = -(yTrue[i,0] - yPred[i,0]) / ((1 - yPred[i,0]) * yPred[i,0]);
        }
        
        return Result;
    }

    public static double BinaryCrossEntropyValue(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows)
        {
            throw new ArgumentException("Number of Batch is not the same!");
        }

        int BatchSize = yPred.Rows;

        double Loss = 0;
        double eps = 1e-12;

        for(int i = 0; i < BatchSize; i++)
        {
            double p = Math.Clamp(yPred[i,0], eps, 1 - eps);
            double y = yTrue[i,0];

            Loss += -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
        }

        return Loss;
    }

    public static Matrix CrossEntropyGrad(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows || yPred.Cols != yTrue.Cols)
        {
            throw new ArgumentException("Shape mismatch");
        }

        int BatchSize = yPred.Rows;
        int numClass = yPred.Cols;

        Matrix Result = new Matrix(BatchSize, numClass);

        for(int i = 0; i < BatchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                double p = Math.Max(yPred[i,j], Utility.e12Eps);
                Result[i,j] = -yTrue[i,j] / p;
            }
        }

        return Result;
    }

    public static double CrossEntropyValue(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows || yPred.Cols != yTrue.Cols)
        {
            throw new ArgumentException("Shape mismatch!");
        }

        int BatchSize = yPred.Rows;
        int numClass = yPred.Cols;

        double loss = 0;

        for(int i = 0; i < BatchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                double p = Math.Max(yPred[i,j], Utility.e12Eps);
                loss += -yTrue[i,j]*Math.Log(p);
            }
        }

        return loss;
    }
}