class Adam : Optimizer
{
    public double Beta1, Beta2;

    public double TBeta1, TBeta2;
    public List<Matrix> Velocities;
    public List<Matrix> Variances;
    public Adam(Sequential sequential, double learningRate, double beta1 = 0.9, double beta2 = 0.999) : base(sequential, learningRate)
    {
        Velocities = new List<Matrix>();
        Variances = new List<Matrix>();

        foreach(Layer layer in sequential.Layers)
        {
            foreach(Parameter parameter in layer.Parameters())
            {
                int paramRows = parameter.Data.Rows;
                int paramCols = parameter.Data.Cols;

                Matrix velocity = new Matrix(paramRows, paramCols);
                Matrix variance = new Matrix(paramRows, paramCols);

                Velocities.Add(velocity);
                Variances.Add(variance);
            }
        }

        Beta1 = TBeta1 = beta1;
        Beta2 = TBeta2 = beta2;
    }

    public override void Step()
    {
        int totalParam = Parameters.Count;

        for(int i = 0; i < totalParam; i++)
        {
            Parameter parameter = Parameters[i];
            Matrix velocity = Velocities[i];
            Matrix variance = Variances[i];

            int paramRows = parameter.Grad.Rows;
            int paramCols = parameter.Grad.Cols;

            for(int j = 0; j < paramRows; j++)
            {
                for(int k = 0; k < paramCols; k++)
                {
                    double grad = parameter.Grad[j,k];
                    velocity[j,k] = Beta1 * velocity[j,k] + (1 - Beta1) * grad;
                    variance[j,k] = Beta2 * variance[j,k] + (1 - Beta2) * grad * grad;

                    double vHat = velocity[j,k] / (1 - TBeta1);
                    double varHat = variance[j,k] / (1 - TBeta2);
                    
                    parameter.Data[j,k] -= LearningRate * vHat / (Math.Sqrt(varHat) + Utility.e5Eps);

                    
                }
            }
        }

        TBeta1 *= TBeta1;
        TBeta2 *= TBeta2;
    }

    public override void ZeroGrad()
    {
        foreach(Parameter parameter in Parameters)
        {
            int paramRows = parameter.Grad.Rows;
            int paramCols = parameter.Grad.Cols;

            for(int i = 0; i < paramRows; i++)
            {
                for(int j = 0; j < paramCols; j++)
                {
                    parameter.Grad[i,j] = 0;
                }
            }
        }
    }
}