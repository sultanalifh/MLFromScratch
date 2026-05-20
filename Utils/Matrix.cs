using System.Text.Json.Serialization;

class Matrix : Tensor
{
    [JsonInclude]
    public int Rows => Shape[0];

    [JsonInclude]
    public int Cols => Shape[1];

    public Matrix(int rows, int cols) : base(rows, cols)
    {
        if(rows <= 0 || cols <= 0)
        {
            throw new ArgumentException("Size must be positive!");
        }
    }
    public double RowSum(int row)
    {
        if(row < 0 || row >= Rows)
        {
            throw new ArgumentOutOfRangeException("Index row out of bound!");
        }

        double Sum = 0;

        for(int i = 0; i < Cols; i++)
        {
            Sum += this[row,i];
        }

        return Sum;
    }

    public override Matrix Clone()
    {
        Matrix clone = new Matrix(Rows, Cols);

        Array.Copy(Data, clone.Data, Data.Length);

        return clone;
    }
}