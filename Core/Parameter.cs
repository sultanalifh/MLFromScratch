class Parameter
{
    public Matrix Data;
    public Matrix Grad;

    public Parameter(Matrix data)
    {
        Data = data;
        Grad = new Matrix(Data.Rows, Data.Cols);
    }

    public Parameter(int rows, int cols)
    {
        Data = new Matrix(rows,cols);
        Grad = new Matrix(rows,cols);
    }
}