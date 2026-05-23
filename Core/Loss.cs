enum ModelLoss
{
    None,
    MSE,
    BCE,
    CE
}

static class Loss
{
    public static double MeanSquarredErrorValue(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape was not match!");
        }
        
        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Cannot compute loss for non-matrices Tensor!");
        }

        int batchSize = predShape[0];
        int numClass = predShape[1];

        double loss = 0;

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                loss += (yTrue[i,j] - yPred[i,j]) * (yTrue[i,j] - yPred[i,j]);
            }
        }

        loss /= batchSize;

        return loss;
    }

    public static Tensor MeanSquarredErrorGrad(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape was not match!");
        }

        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Cannot compute loss for non-matrices Tensor!");
        }

        int batchSize = predShape[0];
        int numClass = predShape[1];

        Tensor lossGrad = new Tensor(batchSize, numClass);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                lossGrad[i,j] += 2 * (yPred[i,j] - yTrue[i,j]);
            }
        }

        return lossGrad;
    }

    public static double BinaryCrossEntropyValue(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape was not match!");
        }

        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Cannot compute loss for non-matrices Tensor!");
        }

        int batchSize = predShape[0];
        int numClass = predShape[1];

        double loss = 0;

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                loss += -(yTrue[i,j]*Math.Log(yPred[i,j]) + (1 - yTrue[i,j])*Math.Log(1 - yPred[i,j]));
            }
        }

        loss /= batchSize;

        return loss;
    }
    
    public static Tensor BinaryCrossEntropyGrad(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape was not match!");
        }

        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Cannot compute loss for non-matrices Tensor");
        }

        int batchSize = predShape[0];
        int numClass = predShape[1];

        Tensor lossGrad = new Tensor(batchSize, numClass);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                lossGrad[i,j] = -(yTrue[i,j] - yPred[i,j]) / ((1 - yPred[i,j]) * yPred[i,j]);
            }
        }

        return lossGrad;
    }
    
    public static double CrossEntropyValue(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape was not match!");
        }

        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Cannot compute loss for non-matrices Tensor!");
        }

        int batchSize = predShape[0];
        int numClass = predShape[1];

        double loss = 0;

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                double p = Math.Max(yPred[i,j], Utility.e12Eps);
                loss += -yTrue[i,j] * Math.Log(p);
            }
        }

        loss /= batchSize;

        return loss;
    }

    public static Tensor CrossEntropyGrad(Tensor yPred, Tensor yTrue)
    {
        if (!yPred.ShapeMatch(yTrue))
        {
            throw new ArgumentException("Shape was not match!");
        }

        int[] predShape = yPred.Shape;

        if(predShape.Length != 2)
        {
            throw new ArgumentException("Cannot compute loss for non-matrices Tensor!");
        }

        int batchSize = predShape[0];
        int numClass = predShape[1];

        Tensor lossGrad = new Tensor(batchSize, numClass);

        for(int i = 0; i < batchSize; i++)
        {
            for(int j = 0; j < numClass; j++)
            {
                double p = Math.Max(yPred[i,j], Utility.e12Eps);
                lossGrad[i,j] += -yTrue[i,j] / p;
            }
        }

        return lossGrad;
    }

    public static double L2Regularization(NeuralNetwork network)
    {
        double l2Loss = 0;

        Sequential sequential = network.Sequential;

        foreach(Parameter parameter in sequential.Parameters())
        {
            string paramName = parameter.Name;

            if(paramName != "Weight")
            {
                continue;
            }

            double[] weights = parameter.Data.Data;

            foreach(double weight in weights)
            {
                l2Loss += network.Lambda * weight * weight;
            }
        }

        return l2Loss;
    }

    public static double LossValue(Tensor yPred, Tensor yTrue, ModelLoss lossType) => lossType switch
    {
        ModelLoss.MSE => MeanSquarredErrorValue(yPred, yTrue),
        ModelLoss.BCE => BinaryCrossEntropyValue(yPred, yTrue),
        ModelLoss.CE => CrossEntropyValue(yPred, yTrue),
        _ => throw new ArgumentException("No such loss type found!")
    };

    public static Tensor LossGrad(Tensor yPred, Tensor yTrue, ModelLoss lossType) => lossType switch
    {
        ModelLoss.MSE => MeanSquarredErrorGrad(yPred, yTrue),
        ModelLoss.BCE => BinaryCrossEntropyGrad(yPred, yTrue),
        ModelLoss.CE => CrossEntropyGrad(yPred, yTrue),
        _ => throw new ArgumentException("No such loss type found!")
    };

}