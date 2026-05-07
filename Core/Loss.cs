static class Loss
{
    public static Matrix BinaryCrossEntropy(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows)
        {
            throw new ArgumentException("Number of predictions was not the same!");
        }

        Matrix Result = new Matrix(yPred.Rows, 1);

        for(int i = 0; i < yPred.Rows; i++)
        {
            Result[i,0] = -(yTrue[i,0] - yPred[i,0]) / ((1 - yPred[i,0]) * yPred[i,0]);
        }

        return Result;
    }
}