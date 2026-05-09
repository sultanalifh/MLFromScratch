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
            throw new ArgumentException("Number of Predictions was not the same!");
        }

        int BatchSize = yPred.Rows;

        double Loss = 0;
        double eps = 1e-12;

        Matrix Result = new Matrix(BatchSize, 1);

        for(int i = 0; i < BatchSize; i++)
        {
            double p = Math.Clamp(yPred[i,0], eps, 1 - eps);
            double y = yTrue[i,0];

            Loss += -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
        }

        return Loss;
    }
}