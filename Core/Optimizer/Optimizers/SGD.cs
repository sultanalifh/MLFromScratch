class SGD : Optimizer
{
    public SGD(Sequential sequential, double learningRate) : base(sequential, learningRate)
    {
        
    }

    public override void Step()
    {
        foreach(Parameter parameter in Parameters)
        {
            int ParamRows = parameter.Data.Rows;
            int ParamCols = parameter.Data.Cols;

            for(int i = 0; i < ParamRows; i++)
            {
                for(int j = 0; j < ParamCols; j++)
                {
                    parameter.Data[i,j] -= parameter.Grad[i,j] * LearningRate;
                }
            }
        }
    }

    public override void ZeroGrad()
    {
        foreach(Parameter parameter in Parameters)
        {
            int ParamRows = parameter.Data.Rows;
            int ParamCols = parameter.Data.Cols;

            for(int i = 0; i < ParamRows; i++)
            {
                for(int j = 0; j < ParamCols; j++)
                {
                    parameter.Grad[i,j] = 0;
                }
            }
        }
    }
}