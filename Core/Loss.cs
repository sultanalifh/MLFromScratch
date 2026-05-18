enum ModelLoss
{
    None,
    MSE,
    BCE,
    CE
}

static class Loss
{

    public static Matrix MeanSquarredErrorGrad(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows || yPred.Cols != yTrue.Cols)
        {
            throw new ArgumentException("Shape mismatch!");
        }
        else if(yPred.Cols != 1)
        {
            throw new ArgumentException("Invalid MSE loss computation for inputSize: " + yPred.Cols);
        }

        int batchSize = yPred.Rows;
        Matrix result = new Matrix(batchSize, 1);

        for(int i = 0; i < batchSize; i++)
        {
            result[i,0] = 2 * (yTrue[i,0] - yPred[i,0]);
        }

        return result;
    }

    public static double MeanSquarredErrorValue(Matrix yPred, Matrix yTrue)
    {
        if(yPred.Rows != yTrue.Rows || yPred.Cols != yTrue.Cols)
        {
            throw new ArgumentException("Shape mismatch!");
        }
        else if(yPred.Cols != 1)
        {
            throw new ArgumentException("Invalid MSE loss computation for inputSize: " + yPred.Cols);
        }
        
        int batchSize = yPred.Rows;
        double loss = 0;

        for(int i = 0; i < batchSize; i++)
        {
            loss += (yTrue[i,0] - yPred[i,0]) * (yTrue[i,0] - yPred[i,0]);
        }

        return loss / batchSize;
    }
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

        return loss / BatchSize;
    }

    public static double L2Regularization(Sequential sequential, double lambda)
    {
        double l2Loss = 0;

        foreach(Layer layer in sequential.Layers)
        {
            foreach(Parameter parameter in layer.Parameters())
            {
                if(parameter.Name != "Weight") continue;

                int weightRows = parameter.Data.Rows;
                int weightCols = parameter.Data.Cols;

                for(int i = 0; i < weightRows; i++)
                {
                    for(int j = 0; j < weightCols; j++)
                    {
                        l2Loss += parameter.Data[i,j] * parameter.Data[i,j];
                    }
                }
            }
        }

        return l2Loss * lambda;
    }

    public static Matrix LossGrad(Matrix yPred, Matrix yTrue, ModelLoss loss) => loss switch
    {
        ModelLoss.MSE => MeanSquarredErrorGrad(yPred, yTrue),
        ModelLoss.BCE => BinaryCrossEntropyGrad(yPred, yTrue),
        ModelLoss.CE => CrossEntropyGrad(yPred, yTrue),
        _ => new Matrix(yPred.Rows, yPred.Cols)
    };

    public static double LossValue(Matrix yPred, Matrix yTrue, ModelLoss loss) => loss switch
    {
        ModelLoss.MSE => MeanSquarredErrorValue(yPred, yTrue),
        ModelLoss.BCE => BinaryCrossEntropyValue(yPred, yTrue),
        ModelLoss.CE => CrossEntropyValue(yPred, yTrue),
        _ => double.NaN,
    };
}