static class Metrics
{
    public static double MultiClassificationAccuracy(Tensor yPred, Tensor yTrue)
    {
        if(!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape Mismatch!");
        }

        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Couldn't compute accuracy for non-matrices Tensor");
        }

        int numPred = predShape[0];
        int numClass = predShape[1];

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