class Momentum : Optimizer
{
    public List<Matrix> Velocities;

    public double momentum;
    public Momentum(Sequential sequential, double learningRate, double momentum) : base(sequential, learningRate)
    {
        Velocities = new List<Matrix>();

        foreach(Parameter parameter in Parameters)
        {
            Matrix velocity = new Matrix(parameter.Data.Rows, parameter.Data.Cols);
            Velocities.Add(velocity);
        }

        this.momentum = momentum;
    }

    public override void Step()
    {
        int paramSize = Parameters.Count;

        for(int i = 0; i < paramSize; i++)
        {
            Parameter parameter = Parameters[i];
            Matrix velocity = Velocities[i];

            int paramRows = parameter.Grad.Rows;
            int paramCols = parameter.Grad.Cols;

            for(int j = 0; j < paramRows; j++)
            {
                for(int k = 0; k < paramCols; k++)
                {
                    velocity[j,k] = momentum * velocity[j,k] + parameter.Grad[j,k] * LearningRate;
                    parameter.Data[j,k] -= velocity[j,k];
                }
            }
        }
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