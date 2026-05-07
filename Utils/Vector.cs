class Vector
{
    private double[] _Data;
    public int Length
    {
        get
        {
            return _Data.Length;
        }
    }

    public double this[int index]
    {

        get
        {
            if(index < 0 || index >= Length)
            {
                throw new ArgumentOutOfRangeException("Index out of range!");
            }
            return _Data[index];
        }
        set
        {
            _Data[index] = value;
        }
    }

    public Vector(int Size)
    {
        if(Size <= 0)
        {
            throw new ArgumentException("Size must be positive");
        }

        _Data = new double[Size];
    }
}