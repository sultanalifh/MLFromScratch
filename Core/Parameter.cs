using System.Text.Json.Serialization;

class Parameter
{
    [JsonInclude]
    public string Name;

    [JsonInclude]
    public Tensor Data;

    public Tensor Grad;
    
    public Parameter(string name, params int[] shape)
    {
        Name = name;
        Data = new Tensor(shape);
        Grad = Data.Clone();
    }

    // Tensor like family compatible

    public Parameter(string name, Tensor tensor)
    {
        Name = name;
        Data = tensor;
        Grad = Data.Clone();
    }
}