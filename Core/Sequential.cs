class Sequential
{
    public List<Layer> Layers;

    public Sequential()
    {
        Layers = new List<Layer>();
    }

    public void Add(Layer layer)
    {
        Layers.Add(layer);
    }
    public Matrix Backward(Matrix yPred, Matrix yTrue)
    {
        yPred = Loss.CrossEntropyGrad(yPred, yTrue);

        for(int i = Layers.Count - 1; i >= 0; i--)
        {
            yPred = Layers[i].Backward(yPred);
        }

        return yPred;
    }
    public Matrix Predict(Matrix x)
    {
        for(int i = 0; i < Layers.Count; i++)
        {
            x = Layers[i].Forward(x);
        }

        return x;
    }
    public IEnumerable<Parameter> Parameters()
    {
        foreach(Layer layer in Layers)
        {
            foreach(Parameter param in layer.Parameters())
            {
                yield return param;
            }
        }
    }

    public void ZeroGrad()
    {
        foreach(Layer layer in Layers)
        {
            foreach(Parameter parameter in layer.Parameters())
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
}