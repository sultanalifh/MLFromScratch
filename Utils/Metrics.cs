static class Metrics
{
    public static double MultiClassificationAccuracy(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows || yPred.Cols != yTrue.Cols)
        {
            throw new ArgumentException("Shape Mismatch!");
        }
        int numPred = yPred.Rows;
        int numClass = yPred.Cols;

        double correct = 0;

        for(int i = 0; i < numPred; i++)
        {
            int maxProb = 0;

            for(int j = 0; j < numClass; j++)
            {
                if(yPred[i,j] > yPred[i, maxProb])
                {
                    maxProb = j;
                }
            }

            if(yTrue[i,maxProb] == 1)
            {
                correct++;
            }
        }

        return correct / numPred;
    }
}