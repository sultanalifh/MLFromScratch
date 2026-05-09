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

    public Matrix Forward(Matrix x)
    {
        for(int i = 0; i < Layers.Count; i++)
        {
            x = Layers[i].Forward(x);
        }

        return x;
    }

    public Matrix Backward(Matrix yPred, Matrix yTrue)
    {
        yPred = Loss.BinaryCrossEntropyGrad(yPred, yTrue);

        for(int i = Layers.Count - 1; i >= 0; i--)
        {
            yPred = Layers[i].Backward(yPred);
        }

        return yPred;
    }

    public void LearnGradient(double learningRate)
    {
        for(int i = 0; i < Layers.Count; i++)
        {
            Layers[i].Step(learningRate);
        }
    }

    public Matrix Predict(Matrix x)
    {
        for(int i = 0; i < Layers.Count; i++)
        {
            x = Layers[i].Forward(x);
        }

        return x;
    }
}