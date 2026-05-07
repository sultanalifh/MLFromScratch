class Neuron
{
    private Vector _Weight;

    public Vector Weight
    {
        get
        {
            return _Weight;
        }
    }

    public Neuron(int Size)
    {
        if(Size <= 0)
        {
            throw new ArgumentException("Size must be positive");
        }
            
        _Weight = new Vector(Size);

        double std = Math.Sqrt(2 / Size);

        for(int i=0;i<Size;i++)
        {
            double u1 = Utility.Random.NextDouble();
            double u2 = Utility.Random.NextDouble();
            double z = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            _Weight[i] = z * std;
        }

    }

    public double Forward(Vector x)
    {
        double z = _Weight.Dot(x) + _Weight.Bias;

        return z;
    }
}