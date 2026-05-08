class Matrix
{
    public int Rows;
    public int Cols;
    private double[,] _Data;

    public double this[int row, int col]
    {
        get
        {
            return _Data[row, col];
        }
        set
        {
            _Data[row, col] = value;
        }
    }

    public Matrix(int rows, int cols)
    {
        if(rows <= 0 || cols <= 0)
        {
            throw new ArgumentException("Size must be positive!");
        }

        Rows = rows;
        Cols = cols;

        _Data = new double[Rows, Cols];
    }

    public Matrix Dot(Matrix other)
    {
        if(other.Rows != Cols)
        {
            throw new ArgumentException("Dot not possible!");
        }

        Matrix Result = new Matrix(Rows, other.Cols);

        for(int i = 0; i < Rows; i++)
        {
            for(int j = 0; j < other.Cols; j++)
            {
                double sum = 0; 

                for(int k = 0; k < Cols; k++)
                {
                    sum += this[i,k] * other[k,j];
                }
                Result[i,j] = sum;
            }
        }

        return Result;
    }

    public Matrix Transpose()
    {
        Matrix Result = new Matrix(Cols, Rows);

        for(int i = 0; i < Rows; i++)
        {
            for(int j = 0; j < Cols; j++)
            {
                Result[j,i] = this[i,j];
            }
        }

        return Result;
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

    public Matrix Clone()
    {
        Matrix clone = new Matrix(Rows, Cols);

        for(int i = 0; i < Rows; i++)
        {
            for(int j = 0; j < Cols; j++)
            {
                clone[i,j] = this[i,j];
            }
        }

        return clone;
    }
}