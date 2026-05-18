using System.Text.Json.Serialization;

class Sequential
{
    [JsonInclude]
    public List<Layer> Layers;

    public Sequential()
    {
        Layers = new List<Layer>();
    }

    public Sequential(List<Layer> layers)
    {
        Layers = layers;
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
    public Matrix Backward(Matrix yPred)
    {
        for(int i = Layers.Count - 1; i >= 0; i--)
        {
            yPred = Layers[i].Backward(yPred);
        }

        return yPred;
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
}